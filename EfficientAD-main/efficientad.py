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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from freia_funcs import *
import torch.nn as nn
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt
import time
# from train_nf_teacher import main_teacher

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

class DefectDataset(Dataset):
    def __init__(self, set='train', get_mask=False):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # TODO make this (bottle) dynamic
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

            if self.get_mask and self.set != 'train' and sc != 'good':
                mask_dir = os.path.join(root, 'ground_truth', sc)
                self.masks.extend(
                    [os.path.join(mask_dir, p) for p in sorted(os.listdir(mask_dir))])

        self.img_mean = torch.FloatTensor([0.485, 0.456, 0.406])[:, None, None]
        self.img_std = torch.FloatTensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        fg = torch.ones([1, 192, 192])

        with open(self.images[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.image_transforms(img)

        image_name = os.path.splitext(os.path.basename(self.images[index]))[0]
        label = self.labels[index]

        ret = [fg, label, img, image_name]

        if self.get_mask and self.set != 'train':
            if label == 0:
                mask = torch.zeros(1, 768, 768)
            else:
                with open(self.masks[index], 'rb') as f:
                    mask = Image.open(f).convert('L')

                mask_transforms = transforms.Compose([
                    transforms.Resize((768, 768)),
                    transforms.ToTensor()
                ])

                mask = mask_transforms(mask)
                mask = (mask > 0.5).float()
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
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == self.layer_idx:
                # Returning features from the specified layer.
                return x

# Function to create a normalizing flow model for the teacher.
def get_nf(input_dim=304, channels_hidden=1024):
    nodes = list()
    # Main input node.
    nodes.append(InputNode(32, name='input'))
    nodes.append(InputNode(input_dim, name='input'))
    kernel_sizes = [3, 3, 3, 5]
    # Creating coupling blocks.
    for k in range(4):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # Conditional coupling layer if positional encoding is used.
        nodes.append(Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                          {'clamp': 3.0,
                           'F_class': F_conv,
                           'cond_dim': 32,
                           'F_args': {'channels_hidden': channels_hidden,
                                      'kernel_size': kernel_sizes[k]}},
                          name=F'conv_{k}'))
    # Output node
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
        inp_feat = 336
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
        # Concatenate positional encoding to the input
        x = torch.cat(x, dim=1)
        # Process input through the initial convolution layer
        x = self.act(self.conv1(x))
        # Pass the output through each residual block
        for i in range(len(self.res)):
            x = self.res[i](x)

        # Final convolution to produce the output feature map
        x = self.conv2(x)
        return x

    def jacobian(self, run_forward=False):
        return [0] # Placeholder for Jacobian computation, not applicable for student and autoencoder models.

# Function to create positional encoding.
def positionalencoding2d(D, H, W):
    """
    taken from https://github.com/gudovskiy/cflow-ad
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    # Creates a positional encoding matrix.
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(np.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    # Returns the positional encoding.
    return P.to('cuda')[None]

def get_autoencoder(out_channels=304):
    return nn.Sequential(
        # encoder
        nn.Conv2d(in_channels=336, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        # decoder
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.AdaptiveAvgPool2d((24, 24))
    )

class StudentTeacherModel(nn.Module):
    def __init__(self, nf=False, n_blocks=4, channels_hidden=64, model_autoencoder=False):
        super(StudentTeacherModel, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.model_autoencoder = model_autoencoder
        if model_autoencoder:
            self.net = get_autoencoder(304)
        elif nf:
            self.net = get_nf()
        else:
            self.net = Student(channels_hidden=channels_hidden, n_blocks=n_blocks)

        self.pos_enc = positionalencoding2d(32, 24, 24)

    def forward(self, x):
        # Feature extraction
        with torch.no_grad():
            inp = self.feature_extractor(x)

        # Processing through the network with positional encoding.
        cond = self.pos_enc.tile(inp.shape[0], 1, 1, 1)

        if self.model_autoencoder:
            ae_input = torch.cat([cond, inp], dim=1)
            return self.net(ae_input)
        else:
            # Passing input through the network
            z = self.net([cond, inp])
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

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    # TODO make this (bottle) dynamic
    test_output_dir = 'output/anomaly_maps/bottle'

    # TODO make this (bottle) dynamic
    teacher = torch.load('./models/teacher_nf_bottle.pth')
    teacher.eval()
    teacher.cuda()

    full_train_set = DefectDataset(set='train', get_mask=False)
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(42)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    train_loader = DataLoader(train_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_set, pin_memory=True, batch_size=8)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=True), pin_memory=True, batch_size=1, shuffle=False, drop_last=False)

    student = StudentTeacherModel(nf=False, channels_hidden=1024, n_blocks=4)
    student.train()
    student.cuda()

    autoencoder = StudentTeacherModel(model_autoencoder=True)
    autoencoder.train()
    autoencoder.cuda()

    train_steps = 1
    optimizer = torch.optim.Adam(itertools.chain(student.net.parameters(), autoencoder.net.parameters()), lr=2e-4, eps=1e-08, weight_decay=1e-5)

    # TODO: try reduce on plateau, cos schedulers
    temp_loss = 150
    temp_loss_general = 200
    train_loss = []
    for sub_epoch in range(train_steps):
        train_loss = list()
        print('Epoch {} out of {}'.format(sub_epoch+1, train_steps))
        for i, data in enumerate(tqdm(train_loader, disable=True)):
            optimizer.zero_grad()
            fg, labels, image, _ = data
            fg, image = [t.to('cuda') for t in [fg, image]]

            fg_down = downsampling(fg, (24, 24), bin=False)

            with torch.no_grad():
                teacher_output_st, _ = teacher(image)

            student_output_st, _ = student(image)
            student_output_st = student_output_st[:, :304]

            distance_st = (teacher_output_st - student_output_st) ** 2
            d_hard = torch.quantile(distance_st, q=0.999)
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])

            # TODO add image penalty from imagenet dataset with 167GB of data
            loss_st = loss_hard

            ae_output = autoencoder(image)
            with torch.no_grad():
                teacher_output_ae, _ = teacher(image)
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

            # TODO: try to implement early-stopping

            print(f'Loss after epoch: {loss_total.item()}')
            train_loss.append(loss_total.item())
            if (loss_total.item() < temp_loss):
                temp_loss = loss_total.item()
                # Try to save the whole model instead of state_dict()
                  TODO make this (bottle) dynamic
                torch.save(student, join('./models', 'student_bottle.pth'))
                torch.save(autoencoder, join('./models', 'autoencoder_bottle.pth'))


        if (temp_loss < temp_loss_general):
            temp_loss_general = temp_loss
            print("Current best loss: {:.4f}  ".format(temp_loss_general))

    save_loss_graph(train_loss, test_output_dir)

    student.eval()
    autoencoder.eval()

    teacher.eval()

    # # TODO make this (bottle) dynamic
    # student = torch.load('./models/student_bottle.pth')
    # student.eval()
    # student.cuda()
    # # TODO make this (bottle) dynamic
    # autoencoder = torch.load('./models/autoencoder_bottle.pth')
    # autoencoder.eval()
    # autoencoder.cuda()

    auc, pixel_roc_auc, image_f1, pixel_f1, image_recall, image_precision, pixel_recall, pixel_precision, latency = test(test_loader=test_loader, teacher=teacher, student=student, autoencoder=autoencoder, test_output_dir=test_output_dir, desc='Final inference')
    print('Final pixel auc: {:.4f}'.format(pixel_roc_auc))
    print('Final image auc: {:.4f}'.format(auc))
    print('Final pixel f1: {:.4f}'.format(pixel_f1))
    print('Final image f1: {:.4f}'.format(image_f1))
    print('Final pixel recall: {:.4f}'.format(pixel_recall))
    print('Final image recall: {:.4f}'.format(image_recall))
    print('Final pixel precision: {:.4f}'.format(pixel_precision))
    print('Final image precision: {:.4f}'.format(image_precision))
    print('Final average processing latency: {:.4f} ms/img'.format(latency))

@torch.no_grad()
def predict(image, teacher, student, autoencoder):
    start_time = time.time()
    # Generate predictions using the teacher model
    teacher_output, _ = teacher(image)
    # Generate predictions using the student model
    student_output, _ = student(image)
    # Generate reconstructions using the autoencoder
    autoencoder_output = autoencoder(image)
    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000
    # Calculate the mean squared error (MSE) between the teacher and student outputs
    map_st = torch.mean((teacher_output - student_output[:, :304])**2, dim=1, keepdim=True)
    # Calculate the MSE between the autoencoder output and the student output
    map_ae = torch.mean((autoencoder_output - student_output[:, 304:])**2, dim=1, keepdim=True)
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae, latency_ms

def test(test_loader, teacher, student, autoencoder, test_output_dir=None, desc='Running inference'):
    y_true = []
    y_score = []
    y_score_binary = []
    mask_flat_combined = []
    map_flat_combined = []
    latencies = []
    mask_save_data = []

    for data in tqdm(test_loader, desc=desc):
        fg, labels, image, image_name, mask = data
        _, C, H, W = image.shape
        orig_width = W
        orig_height = H

        # TODO make this dynamic
        defect_classes = ["good", "broken_large", "broken_small", "contamination"]

        fg, image, mask = [t.to('cuda') for t in [fg, image, mask]]

        map_combined, map_st, map_ae, latency_ms = predict(image=image, teacher=teacher, student=student, autoencoder=autoencoder)
        map_combined = torch.nn.functional.pad(map_combined, (4, 4, 4, 4))
        map_combined = torch.nn.functional.interpolate(
            map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        map_combined = 1 / (1 + np.exp(-map_combined))

        latencies.append(latency_ms)

        defect_class = defect_classes[labels.item()]
        if test_output_dir is not None:
            img_nm = image_name[0]
            mask_save_data.append((os.path.join(test_output_dir, defect_class), os.path.join(test_output_dir, defect_class, img_nm + '.png'), map_combined))

            #Saving images heatmap
            # TODO make this (bottle) dynamic
            test_output_dir_heat = 'output/anomaly_maps/bottle_heat'
            if not os.path.exists(os.path.join(test_output_dir_heat, defect_class)):
                os.makedirs(os.path.join(test_output_dir_heat, defect_class))
            dpi = 100
            fig_size = 768 / dpi
            fig = plt.figure(figsize=(fig_size, fig_size), dpi=dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off()
            heatmap = ax.imshow(map_combined, cmap='hot', interpolation='nearest')
            plt.axis('off')
            fig.savefig(os.path.join(test_output_dir_heat, defect_class, img_nm + '.png'), dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

        y_true_image = 0 if defect_class == "good" else 1
        y_score_image = np.max(map_combined)
        y_true.append(y_true_image)
        y_score.append(y_score_image)

        mask_flat = mask.flatten().cpu().numpy()
        map_flat = map_combined.flatten()
        mask_flat_combined.extend(mask_flat)
        map_flat_combined.extend(map_flat)

    # pixel-level auc calculations
    pixel_roc_auc = roc_auc_score(y_true=mask_flat_combined, y_score=map_flat_combined)
    pixel_roc_auc = pixel_roc_auc * 100
    # image-level auc calculations
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    auc = auc * 100

    image_f1_threshold, pixel_f1_threshold = save_curves(map_flat_combined, mask_flat_combined, y_score, y_true, auc, pixel_roc_auc, test_output_dir)

    # Saving predicted masks as png with the best calculated threshold:
    save_predicted_masks(mask_save_data, pixel_f1_threshold)

    # Convert image and pixel predictions to binary with optimap thresholds
    y_score_binary = (y_score > image_f1_threshold).astype(np.float32)
    map_flat_combined_binary = (map_flat_combined > pixel_f1_threshold).astype(np.float32)

    # image-level f1 calculations
    image_f1 = f1_score(y_true=y_true, y_pred=y_score_binary)
    # pixel-level f1 calculations
    pixel_f1 = f1_score(y_true=mask_flat_combined, y_pred=map_flat_combined_binary)

    # image-level precision and recall calculations
    image_recall = recall_score(y_true, y_score_binary)
    image_precision = precision_score(y_true, y_score_binary)
    # pixel-level precision and recall calculations
    pixel_recall = recall_score(mask_flat_combined, map_flat_combined_binary)
    pixel_precision = precision_score(mask_flat_combined, map_flat_combined_binary)

    average_latency = sum(latencies) / len(latencies)
    return auc, pixel_roc_auc, image_f1 * 100, pixel_f1 * 100, image_recall * 100, image_precision * 100, pixel_recall * 100, pixel_precision * 100, average_latency

def save_predicted_masks(save_data, threshold):
    for directory_path, file_path, mask in save_data:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        mask = (mask > threshold).astype(np.float32)
        image_to_save = Image.fromarray((mask * 255).astype(np.uint8))
        image_to_save.save(file_path)

def save_curves(pixel_prediction, pixel_gt, image_predictions, image_gt, image_auc, pixel_auc, location):
    # For pixel-level ROC curve
    fpr_pixel, tpr_pixel, thresholds_pixel = roc_curve(y_true=pixel_gt, y_score=pixel_prediction)
    # For image-level ROC curve
    fpr, tpr, thresholds = roc_curve(y_true=image_gt, y_score=image_predictions)
    # For pixel-level precision-recall curve
    precision_pixel, recall_pixel, thresholds_f1_pixel = precision_recall_curve(pixel_gt, pixel_prediction)
    # For image-level precision-recall curve
    precision, recall, thresholds_f1 = precision_recall_curve(image_gt, image_predictions)

    # Optimal threshold for image f1 calculation
    f1_scores = []
    for p, r in zip(precision, recall):
        if (p + r) != 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0  # Define F1 as 0 if both precision and recall are zero
        f1_scores.append(f1)

    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds_f1[optimal_idx]
    print(f"optimal image f1 threshold: {optimal_threshold}")

    # Optimal threshold for pixel f1 calculation
    f1_scores_pixel = []
    for p, r in zip(precision_pixel, recall_pixel):
        if (p + r) != 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0  # Define F1 as 0 if both precision and recall are zero
        f1_scores_pixel.append(f1)

    optimal_idx_pixel = np.argmax(f1_scores_pixel)
    optimal_threshold_pixel = thresholds_f1_pixel[optimal_idx_pixel]
    print(f"optimal pixel f1 threshold: {optimal_threshold_pixel}")

    if not os.path.exists(os.path.join(location, 'graphs')):
        os.makedirs(os.path.join(location, 'graphs'))

    # Plotting the image-level Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Image-Level')
    plt.legend(loc="best")
    plt.grid(True)
    pr_path = os.path.join(location, 'graphs', 'precision_recall_image.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting the pixel-level Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall_pixel, precision_pixel, label='Precision-Recall Curve', color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Pixel-Level')
    plt.legend(loc="best")
    plt.grid(True)
    pr_pixel_path = os.path.join(location, 'graphs', 'precision_recall_pixel.png')
    plt.savefig(pr_pixel_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting and saving the pixel-level ROC curve
    plt.figure()
    plt.plot(fpr_pixel, tpr_pixel, color='darkorange', lw=2, label='Pixel ROC curve (area = %0.2f)' % pixel_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Pixel-level Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    pixel_roc_path = os.path.join(location, 'graphs', 'pixel_roc.png')
    plt.savefig(pixel_roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plotting and saving the image-level ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', lw=2, label='Overall ROC curve (area = %0.2f)' % image_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Image-level Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    image_roc_path = os.path.join(location, 'graphs', 'image_roc.png')
    plt.savefig(image_roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Pixel-level ROC curve image saved to '{pixel_roc_path}'.")
    print(f"Image-level ROC curve image saved to '{image_roc_path}'.")
    print(f"Pixel-level precision-recall curve image saved to '{pr_pixel_path}'.")
    print(f"Image-level precision-recall curve image saved to '{pr_path}'.")

    return optimal_threshold, optimal_threshold_pixel

def save_loss_graph(train_loss, location):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(location, 'graphs', 'training_loss.png')
    plt.savefig(loss_path)
    plt.close()

    print(f"Training loss curve image saved to '{loss_path}'.")

if __name__ == '__main__':
    main()
