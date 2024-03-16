#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import argparse
import itertools
import os
import random
from tqdm import tqdm
from common import get_autoencoder, get_pdn_small, get_pdn_medium, \
    ImageFolderWithoutTarget, ImageFolderWithPath, InfiniteDataloader
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from freia_funcs import *
import torch.nn as nn
from PIL import Image
from os.path import join
from scipy.ndimage.morphology import binary_dilation
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from IPython.display import display
from torchvision.utils import save_image

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='mvtec_ad',
                        choices=['mvtec_ad', 'mvtec_loco'])
    parser.add_argument('-s', '--subdataset', default='bottle',
                        help='One of 15 sub-datasets of Mvtec AD or 5' +
                             'sub-datasets of Mvtec LOCO')
    parser.add_argument('-o', '--output_dir', default='output/1')
    parser.add_argument('-m', '--model_size', default='small',
                        choices=['small', 'medium'])
    parser.add_argument('-w', '--weights', default='models/teacher_small.pth')
    parser.add_argument('-i', '--imagenet_train_path',
                        default='none',
                        help='Set to "none" to disable ImageNet' +
                             'pretraining penalty. Or see README.md to' +
                             'download ImageNet and set to ImageNet path')
    parser.add_argument('-a', '--mvtec_ad_path',
                        default='./mvtec_anomaly_detection',
                        help='Downloaded Mvtec AD dataset')
    parser.add_argument('-b', '--mvtec_loco_path',
                        default='./mvtec_loco_anomaly_detection',
                        help='Downloaded Mvtec LOCO dataset')
    parser.add_argument('-t', '--train_steps', type=int, default=70000)
    return parser.parse_args()

# constants
seed = 42
on_gpu = torch.cuda.is_available()
out_channels = 384
image_size = 256

# data loading
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
transform_ae = transforms.RandomChoice([
    transforms.ColorJitter(brightness=0.2),
    transforms.ColorJitter(contrast=0.2),
    transforms.ColorJitter(saturation=0.2)
])

def train_transform(image):
    return default_transform(image), default_transform(transform_ae(image))

class DefectDataset(Dataset):
    def __init__(self, set='train', get_mask=True, get_features=True):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.get_features = get_features
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        root = join('./mvtec_anomaly_detection/', 'bottle')
        set_dir = os.path.join(root, set)
        subclass = os.listdir(set_dir)
        subclass.sort()
        class_counter = 1
        for sc in subclass:
            if sc == 'good':
                label = 0
            else:
                label = class_counter
                self.class_names.append(sc)
                class_counter += 1
            sub_dir = os.path.join(set_dir, sc)
            img_dir = sub_dir
            img_paths = os.listdir(img_dir)
            img_paths.sort()
            for p in img_paths:
                i_path = os.path.join(img_dir, p)
                if not i_path.lower().endswith(
                        ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')):
                    continue
                self.images.append(i_path)
                self.labels.append(label)


        self.img_mean = torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
        self.img_std = torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        fg = torch.ones([1, 192, 192])

        if self.set == 'test' or not self.get_features:
            with open(self.images[index], 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = self.image_transforms(img)
        else:
            img = 0

        image_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        label = self.labels[index]

        ret = [fg, label, img, image_name]
        return ret

#TODO revisit to see if it is still needed
class FeatureExtractor(nn.Module):
    def __init__(self, layer_idx=35):
        super(FeatureExtractor, self).__init__()
        self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        # Index of the layer to extract features from.
        self.layer_idx = layer_idx

    # Processing through EfficientNet up to specified layer.
    def forward(self, x):
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.layer_idx:
                # Returning features from the specified layer.
                return x

# Function to create a normalizing flow model for the teacher.
def get_nf(input_dim=304, channels_hidden=64):
    nodes = list()
    # Main input node.
    nodes.append(InputNode(input_dim, name='input'))
    # Creating coupling blocks.
    kernel_sizes = [3, 3, 3, 5]
    for k in range(4):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # Conditional coupling layer if positional encoding is used.
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer_cond,
                          {'clamp': 1.9,
                           'F_class': F_conv,
                           'F_args': {'channels_hidden': channels_hidden,
                                      'kernel_size': kernel_sizes[k]}},
                          name=F'conv_{k}'))
    # Output node.
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    # Creating the reversible graph net.
    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf

class res_block(nn.Module):
    def __init__(self, channels):
        super(res_block, self).__init__()
        # First convolution layer to process the input tensor
        self.l1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Second convolution layer for further processing
        self.l2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # Activation function to introduce non-linearity
        self.act = nn.LeakyReLU()
        # Batch normalization to stabilize and speed up training
        self.bn1 = nn.BatchNorm2d(channels)
        # Second batch normalization for the second convolution layer
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Store the original input for the residual connection
        inp = x
        # First layer processing
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act(x)

        # Second layer processing
        x = self.l2(x)
        x = self.bn2(x)
        x = self.act(x)
        # Adding the input back to the output (residual connection)
        x = x + inp
        return x

# The student model consisting of convolutional layers and residual blocks.
class Student(nn.Module):
    def __init__(self, channels_hidden=1024, n_blocks=4):
        super(Student, self).__init__()
        # Calculate input features, adjust for positional encoding if used
        inp_feat = 304
        # Initial convolution layer to adapt the input feature size
        self.conv1 = nn.Conv2d(inp_feat, channels_hidden, kernel_size=3, padding=1)
        # Final convolution layer to produce the output feature map
        self.conv2 = nn.Conv2d(channels_hidden, 608, kernel_size=3, padding=1)
        # Initialize the residual blocks
        self.res = list()
        # Initializing residual blocks.
        for _ in range(n_blocks):
            self.res.append(res_block(channels_hidden))
        self.res = nn.ModuleList(self.res)
        # Learnable scaling parameter for the output
        self.gamma = nn.Parameter(torch.zeros(1))
        # Activation function for non-linearity
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # Process input through the initial convolution layer
        x = self.act(self.conv1(x))
        # Pass the output through each residual block
        for i in range(len(self.res)):
            x = self.res[i](x)

        # Final convolution to produce the output feature map
        x = self.conv2(x)
        return x

    def jacobian(self, run_forward=False):
        return [0] # Placeholder for Jacobian computation, not applicable for student model.

class StudentTeacherModel(nn.Module):
    def __init__(self, nf=False, n_blocks=4, channels_hidden=64):
        super(StudentTeacherModel, self).__init__()

        self.feature_extractor = FeatureExtractor()
        if nf:
            self.net = get_nf()
        else:
            self.net = Student(channels_hidden=channels_hidden, n_blocks=n_blocks)

    def forward(self, x):
        # Feature extraction based on the mode and configuration.
        # TODO figure out why feature extractor is used and not the image itself
        with torch.no_grad():
            f = self.feature_extractor(x)

        inp = f

        # Processing through the network with positional encoding.
        z = self.net(inp)

        # Calculating the Jacobian for the normalizing flow.
        jac = self.net.jacobian(run_forward=False)[0]
        # Returning the transformed input and Jacobian.
        return z, jac

def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to('cuda')
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down

# TODO: for final result, include image penalty from imagenet dataset with 167GB of data
# TODO: use feature extractor for autoencoder
# TODO: use training augmentations
# TODO: check if feature extractor is actually needed for student network
def main():
    teacher = StudentTeacherModel(nf=True)
    teacher.net.load_state_dict(torch.load('./models/teacher_nf_bottle.pth'))
    teacher.eval()
    teacher.cuda()
    #TODO split train dataset into training and validation
    full_train_set = DefectDataset(set='train', get_mask=False, get_features=False)
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(seed)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    train_loader = DataLoader(train_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_set, pin_memory=True, batch_size=8)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=False, get_features=False), pin_memory=True, batch_size=1, shuffle=False, drop_last=False)
    student = StudentTeacherModel(nf=False, channels_hidden=608, n_blocks=4)
    student.train()
    student.cuda()

    autoencoder = get_autoencoder(304)
    autoencoder.train()
    autoencoder.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    train_steps = 2
    optimizer = torch.optim.Adam(itertools.chain(student.net.parameters(),
                                                 autoencoder.parameters()), lr=2e-4, eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * train_steps), gamma=0.1)

    for sub_epoch in range(train_steps):
        train_loss = list()
        for i, data in enumerate(tqdm(train_loader, disable=False)):
            optimizer.zero_grad()
            fg, labels, image, _ = data
            fg, image = [t.to('cuda') for t in [fg, image]]

            # TODO revisit calculation of student loss using fg_down
            # fg_down = downsampling(fg, (24, 24), bin=False)

            with torch.no_grad():
                teacher_output_st, _ = teacher(image)
                teacher_output_st = (teacher_output_st - teacher_mean) / teacher_std
            student_output_st, _ = student(image)
            student_output_st = student_output_st[:, :304]

            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            # TODO add image penalty
            loss_st = loss_hard

            ae_output = autoencoder(image)
            with torch.no_grad():
                teacher_output_ae, _ = teacher(image)
                teacher_output_ae = (teacher_output_ae - teacher_mean) / teacher_std
            student_output_ae, _ = student(image)
            student_output_ae = student_output_ae[:, 304:]
            distance_ae = (teacher_output_ae - ae_output) ** 2
            distance_stae = (ae_output - student_output_ae) ** 2
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            loss_total = loss_st + loss_ae + loss_stae

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            scheduler.step()

            print("Current loss: {:.4f}  ".format(loss_total.item()))

    # TODO replace with validation loader
    q_st_start, q_st_end, q_ae_start, q_ae_end = map_normalization(
        validation_loader=validation_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, desc='Final map normalization')
    test_output_dir = 'output/anomaly_maps/bottle'
    auc = test(
        test_loader=test_loader, teacher=teacher, student=student,
        autoencoder=autoencoder, teacher_mean=teacher_mean,
        teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
        q_ae_start=q_ae_start, q_ae_end=q_ae_end,
        test_output_dir=test_output_dir, desc='Final inference')
    print('Final image auc: {:.4f}'.format(auc))
def display_image(images, defect_class, image_name):
    # Define the root directory for results
    root_dir = "test_images"
    # Construct the full path for the class-specific directory
    class_dir = os.path.join(root_dir, defect_class)

    # Check if the root directory exists, if not, create it
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Check if the class-specific directory exists, if not, create it
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Define the full path for the image to be saved
    image_path = os.path.join(class_dir, f"{image_name}.png")

    # Since the image tensor is expected to be [1, 3, H, W], remove the batch dimension
    image_tensor = images.squeeze(0)

    # Save the image
    save_image(image_tensor, image_path)

    print(f"Image saved to {image_path}")

def test(test_loader, teacher, student, autoencoder, teacher_mean, teacher_std,
         q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
         desc='Running inference'):
    y_true = []
    y_score = []
    for data in tqdm(test_loader, desc=desc):
        fg, labels, image, image_name = data
        _, C, H, W = image.shape
        orig_width = W
        orig_height = H

        defect_classes = ["good", "broken_large", "broken_small", "contamination"]
        display_image(image, defect_classes[labels.item()], image_name[0])

        fg, image = [t.to('cuda') for t in [fg, image]]

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
            q_ae_start=q_ae_start, q_ae_end=q_ae_end)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

        defect_class = defect_classes[labels.item()]
        if test_output_dir is not None:
            # TODO: make this the name of original image
            img_nm = image_name[0]
            if not os.path.exists(os.path.join(test_output_dir, defect_class)):
                os.makedirs(os.path.join(test_output_dir, defect_class))
            file = os.path.join(test_output_dir, defect_class, img_nm + '.png')
            # TODO consider using .tiff as it is more detailed for pixels
            image_to_save = Image.fromarray((map_combined * 255).astype(np.uint8))
            image_to_save.save(file)

        y_true_image = 0 if defect_class == "good" else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    return auc * 100

@torch.no_grad()
def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
            q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
    teacher_output, _ = teacher(image)
    teacher_output = (teacher_output - teacher_mean) / teacher_std
    student_output, _ = student(image)
    autoencoder_output = autoencoder(image)
    map_st = torch.mean((teacher_output - student_output[:, :304])**2,
                        dim=1, keepdim=True)
    map_ae = torch.mean((autoencoder_output -
                         student_output[:, 304:])**2,
                        dim=1, keepdim=True)
    if q_st_start is not None:
        map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
    if q_ae_start is not None:
        map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae

@torch.no_grad()
def map_normalization(validation_loader, teacher, student, autoencoder,
                      teacher_mean, teacher_std, desc='Map normalization'):
    maps_st = []
    maps_ae = []
    # ignore augmented ae image
    for data in tqdm(validation_loader, desc=desc):
        fg, labels, image, _ = data
        fg, image = [t.to('cuda') for t in [fg, image]]

        map_combined, map_st, map_ae = predict(
            image=image, teacher=teacher, student=student,
            autoencoder=autoencoder, teacher_mean=teacher_mean,
            teacher_std=teacher_std)
        maps_st.append(map_st)
        maps_ae.append(map_ae)
    maps_st = torch.cat(maps_st)
    maps_ae = torch.cat(maps_ae)
    q_st_start = torch.quantile(maps_st, q=0.9)
    q_st_end = torch.quantile(maps_st, q=0.995)
    q_ae_start = torch.quantile(maps_ae, q=0.9)
    q_ae_end = torch.quantile(maps_ae, q=0.995)
    return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for i, data in enumerate(tqdm(train_loader, desc='Computing mean of features')):
        fg, labels, image, _ = data
        fg, image = [t.to('cuda') for t in [fg, image]]

        teacher_output, _ = teacher(image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for i, data in enumerate(tqdm(train_loader, desc='Computing std of features')):
        fg, labels, image, _ = data
        fg, image = [t.to('cuda') for t in [fg, image]]

        teacher_output, _ = teacher(image)
        distance = (teacher_output - channel_mean) ** 2
        mean_distance = torch.mean(distance, dim=[0, 2, 3])
        mean_distances.append(mean_distance)
    channel_var = torch.mean(torch.stack(mean_distances), dim=0)
    channel_var = channel_var[None, :, None, None]
    channel_std = torch.sqrt(channel_var)

    return channel_mean, channel_std

if __name__ == '__main__':
    main()
