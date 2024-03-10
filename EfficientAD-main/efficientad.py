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
        self.depths = list()
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
                if self.set == 'test' and self.get_mask:
                    extension = '_mask' if sc != 'good' else ''
                    mask_path = os.path.join(root, 'ground_truth', sc, p[:-4] + extension + p[-4:])
                    self.masks.append(mask_path)


        self.img_mean = torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
        self.img_std = torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.images)

    def transform(self, x, img_len, binary=False):
        x = x.copy()
        x = torch.FloatTensor(x)
        if len(x.shape) == 2:
            x = x[None, None]
            channels = 1
        elif len(x.shape) == 3:
            x = x.permute(2, 0, 1)[None]
            channels = x.shape[1]
        else:
            raise Exception(f'invalid dimensions of x:{x.shape}')

        x = downsampling(x, (img_len, img_len), bin=binary)
        x = x.reshape(channels, img_len, img_len)
        return x

    def get_3D(self, index):
        sample = np.load(self.depths[index])
        depth = sample[:, :, 0]
        fg = sample[:, :, -1]
        mean_fg = np.sum(fg * depth) / np.sum(fg)
        depth = fg * depth + (1 - fg) * mean_fg
        depth = (depth - mean_fg) * 100
        return depth, fg

    def __getitem__(self, index):
        depth = torch.zeros([1, 192, 192])
        fg = torch.ones([1, 192, 192])

        if self.set == 'test' or not self.get_features:
            with open(self.images[index], 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = self.image_transforms(img)
        else:
            img = 0

        label = self.labels[index]
        feat = self.features[index] if self.get_features else 0

        ret = [depth, fg, label, img, feat]

        if self.set == 'test' and self.get_mask:
            with open(self.masks[index], 'rb') as f:
                mask = Image.open(f)
                mask = self.transform(np.array(mask), 192, binary=True)[:1]
                mask[mask > 0] = 1
                ret.append(mask)
        return ret

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
        # Positional encoding initialization if enabled.

        # Unshuffle operation for processing depth information.
        self.unshuffle = nn.PixelUnshuffle(8)

    def forward(self, x):
        # Feature extraction based on the mode and configuration.
        with torch.no_grad():
            f = self.feature_extractor(x)

        inp = f

        # Processing through the network with positional encoding.
        z = self.net(inp)

        # Calculating the Jacobian for the normalizing flow.
        jac = self.net.jacobian(run_forward=False)[0]
        # Returning the transformed input and Jacobian.
        return z, jac

class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, name, percentage=True):
        self.name = name
        self.max_epoch = 0
        self.best_score = None
        self.last_score = None
        self.percentage = percentage

    def update(self, score, epoch, print_score=False):
        if self.percentage:
            score = score * 100
        self.last_score = score
        improved = False
        if epoch == 0 or score > self.best_score:
            self.best_score = score
            improved = True
        if print_score:
            self.print_score()
        return improved

    def print_score(self):
        print('{:s}: \t last: {:.2f} \t best: {:.2f}'.format(self.name, self.last_score, self.best_score))

def downsampling(x, size, to_tensor=False, bin=True):
    if to_tensor:
        x = torch.FloatTensor(x).to('cuda')
    down = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
    if bin:
        down[down > 0] = 1
    return down

def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None

def get_nf_loss(z, jac, mask=None, per_sample=False, per_pixel=False):
    mask = 0 * mask + 1
    loss_per_pixel = (0.5 * torch.sum(mask * z ** 2, dim=1) - jac * mask[:, 0])
    if per_pixel:
        return loss_per_pixel
    loss_per_sample = torch.mean(loss_per_pixel, dim=(-1, -2))
    if per_sample:
        return loss_per_sample
    return loss_per_sample.mean()

def main():
    # ast teacher model training and validation ---------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pretrain_penalty = False

    train_loader = DataLoader(DefectDataset(set='train', get_mask=False, get_features=False), pin_memory=True,
                                 batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=False, get_features=False), pin_memory=True,
                                batch_size=16, shuffle=False, drop_last=False)

    teacher = StudentTeacherModel(nf=True)
    teacher.cuda()
    optimizer = torch.optim.Adam(teacher.net.parameters(), lr=2e-4, eps=1e-08, weight_decay=1e-5)
    # Observers to track AUROC scores during training.
    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    for epoch in range(5):
        teacher.train()
        print(F'\nTrain epoch {epoch}')
        for sub_epoch in range(5):
            train_loss = list()
            for i, data in enumerate(tqdm(train_loader, disable=False)):
                # Clear gradients.
                optimizer.zero_grad()

                # Unpack data and move to device.
                depth, fg, labels, image, features = data
                depth, fg, labels, image, features = [t.to('cuda') for t in [depth, fg, labels, image, features]]

                # Downsample foreground mask to match the model output size.
                fg_down = downsampling(fg, (24, 24), bin=False)
                # Forward pass through the model.
                z, jac = teacher(image)

                # Calculate loss and backpropagate.
                loss = get_nf_loss(z, jac, fg_down)
                # Convert tensor loss to numpy and store.
                train_loss.append(t2np(loss))

                # Compute gradients.
                loss.backward()
                # Update model parameters.
                optimizer.step()

            # Calculate mean training loss for the epoch.
            mean_train_loss = np.mean(train_loss)
            print('Epoch: {:d}.{:d} \t teacher train loss: {:.4f}'.format(epoch, sub_epoch, mean_train_loss))

    teacher.eval()
    test_loss = list()
    test_labels = list()
    img_nll = list()
    max_nlls = list()

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader, disable=False)):
            # Unpack and move data to device, similar to training phase.
            depth, fg, labels, image, features = data
            depth, fg, image, features = [t.to('cuda') for t in [depth, fg, image, features]]

            fg_down = downsampling(fg, (24, 24), bin=False)
            z, jac = teacher(image)
            # Calculate loss for each sample.
            loss = get_nf_loss(z, jac, fg_down, per_sample=True)
            # Calculate per-pixel loss.
            nll = get_nf_loss(z, jac, fg_down, per_pixel=True)

            img_nll.append(t2np(loss))
            # Track max loss over all pixels.
            max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
            # Calculate mean test loss.
            test_loss.append(loss.mean().item())
            test_labels.append(labels)

    img_nll = np.concatenate(img_nll)
    max_nlls = np.concatenate(max_nlls)
    test_loss = np.mean(np.array(test_loss))

    print('Epoch: {:d} \t teacher test_loss: {:.4f}'.format(1, test_loss))

    test_labels = np.concatenate(test_labels)
    # Prepare anomaly labels.
    is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])
    # TODO: Add roc auc calculations, save the model, return roc auc calculations
    # TODO: Add calculations for mean and max scores from utils.train_dataset function
    # ast student model training and validation ---------------------------
    train_loader = DataLoader(DefectDataset(set='train', get_mask=False, get_features=False), pin_memory=True,
                                 batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=False, get_features=False), pin_memory=True,
                                batch_size=16, shuffle=False, drop_last=False)
    student = StudentTeacherModel(nf=False, channels_hidden=608, n_blocks=4)
    student.train()
    student.cuda()

    autoencoderv2 = get_autoencoder(304)
    autoencoderv2.train()
    autoencoderv2.cuda()

    teacher_mean, teacher_std = teacher_normalization(teacher, train_loader)

    train_steps = 11
    # TODO: load a created teacher
    optimizer = torch.optim.Adam(itertools.chain(student.net.parameters(),
                                                 autoencoderv2.parameters()), lr=2e-4, eps=1e-08, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.95 * train_steps), gamma=0.1)

    for sub_epoch in range(train_steps):
        train_loss = list()
        for i, data in enumerate(tqdm(train_loader, disable=False)):
            optimizer.zero_grad()
            depth, fg, labels, image, features = data
            depth, fg, image, features = [t.to('cuda') for t in [depth, fg, image, features]]

            # TODO revisit calculation of student loss using fg_down
            fg_down = downsampling(fg, (24, 24), bin=False)

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

            ae_output = autoencoderv2(image)
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

# def test(test_set, teacher, student, autoencoder, teacher_mean, teacher_std,
#          q_st_start, q_st_end, q_ae_start, q_ae_end, test_output_dir=None,
#          desc='Running inference'):
#     y_true = []
#     y_score = []
#     for image, target, path in tqdm(test_set, desc=desc):
#         orig_width = image.width
#         orig_height = image.height
#         image = default_transform(image)
#         image = image[None]
#         if on_gpu:
#             image = image.cuda()
#         map_combined, map_st, map_ae = predict(
#             image=image, teacher=teacher, student=student,
#             autoencoder=autoencoder, teacher_mean=teacher_mean,
#             teacher_std=teacher_std, q_st_start=q_st_start, q_st_end=q_st_end,
#             q_ae_start=q_ae_start, q_ae_end=q_ae_end)
#         map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
#         map_combined = torch.nn.functional.interpolate(
#             map_combined, (orig_height, orig_width), mode='bilinear')
#         map_combined = map_combined[0, 0].cpu().numpy()
#
#         defect_class = os.path.basename(os.path.dirname(path))
#         if test_output_dir is not None:
#             img_nm = os.path.split(path)[1].split('.')[0]
#             if not os.path.exists(os.path.join(test_output_dir, defect_class)):
#                 os.makedirs(os.path.join(test_output_dir, defect_class))
#             file = os.path.join(test_output_dir, defect_class, img_nm + '.tiff')
#             tifffile.imwrite(file, map_combined)
#
#         y_true_image = 0 if defect_class == 'good' else 1
#         y_score_image = np.max(map_combined)
#         y_true.append(y_true_image)
#         y_score.append(y_score_image)
#     auc = roc_auc_score(y_true=y_true, y_score=y_score)
#     return auc * 100

# @torch.no_grad()
# def predict(image, teacher, student, autoencoder, teacher_mean, teacher_std,
#             q_st_start=None, q_st_end=None, q_ae_start=None, q_ae_end=None):
#     teacher_output = teacher(image)
#     teacher_output = (teacher_output - teacher_mean) / teacher_std
#     student_output = student(image)
#     autoencoder_output = autoencoder(image)
#     map_st = torch.mean((teacher_output - student_output[:, :out_channels])**2,
#                         dim=1, keepdim=True)
#     map_ae = torch.mean((autoencoder_output -
#                          student_output[:, out_channels:])**2,
#                         dim=1, keepdim=True)
#     if q_st_start is not None:
#         map_st = 0.1 * (map_st - q_st_start) / (q_st_end - q_st_start)
#     if q_ae_start is not None:
#         map_ae = 0.1 * (map_ae - q_ae_start) / (q_ae_end - q_ae_start)
#     map_combined = 0.5 * map_st + 0.5 * map_ae
#     return map_combined, map_st, map_ae
#
# @torch.no_grad()
# def map_normalization(validation_loader, teacher, student, autoencoder,
#                       teacher_mean, teacher_std, desc='Map normalization'):
#     maps_st = []
#     maps_ae = []
#     # ignore augmented ae image
#     for image, _ in tqdm(validation_loader, desc=desc):
#         if on_gpu:
#             image = image.cuda()
#         map_combined, map_st, map_ae = predict(
#             image=image, teacher=teacher, student=student,
#             autoencoder=autoencoder, teacher_mean=teacher_mean,
#             teacher_std=teacher_std)
#         maps_st.append(map_st)
#         maps_ae.append(map_ae)
#     maps_st = torch.cat(maps_st)
#     maps_ae = torch.cat(maps_ae)
#     q_st_start = torch.quantile(maps_st, q=0.9)
#     q_st_end = torch.quantile(maps_st, q=0.995)
#     q_ae_start = torch.quantile(maps_ae, q=0.9)
#     q_ae_end = torch.quantile(maps_ae, q=0.995)
#     return q_st_start, q_st_end, q_ae_start, q_ae_end

@torch.no_grad()
def teacher_normalization(teacher, train_loader):

    mean_outputs = []
    for i, data in enumerate(tqdm(train_loader, desc='Computing mean of features')):
        depth, fg, labels, image, features = data
        depth, fg, image, features = [t.to('cuda') for t in [depth, fg, image, features]]

        teacher_output, _ = teacher(image)
        mean_output = torch.mean(teacher_output, dim=[0, 2, 3])
        mean_outputs.append(mean_output)
    channel_mean = torch.mean(torch.stack(mean_outputs), dim=0)
    channel_mean = channel_mean[None, :, None, None]

    mean_distances = []
    for i, data in enumerate(tqdm(train_loader, desc='Computing std of features')):
        depth, fg, labels, image, features = data
        depth, fg, image, features = [t.to('cuda') for t in [depth, fg, image, features]]

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
