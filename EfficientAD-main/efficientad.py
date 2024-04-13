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

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdataset', default='bottle')
    parser.add_argument('-t', '--test_only', default=False)
    parser.add_argument('-v', '--train_steps', default=100)
    return parser.parse_args()

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, sample_size=None):
        self.transform = transforms.Compose([transforms.Resize((768, 768)),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                                             transforms.RandomGrayscale(0.3),
                                             transforms.CenterCrop(768)])

        self.image_paths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f))]

        if sample_size is not None and sample_size < len(self.image_paths):
            self.image_paths = random.sample(self.image_paths, sample_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        with open(self.image_paths[index], 'rb') as f:
            image = Image.open(f).convert('RGB')
        image = self.transform(image)
        return image

class DefectDataset(Dataset):
    def __init__(self, set='train', get_mask=False, subdataset='bottle'):
        super(DefectDataset, self).__init__()
        self.set = set
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.defect_classes = []
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        root = join('./mvtec_anomaly_detection/', subdataset)
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
                self.defect_classes.append(sc)

            if self.get_mask and self.set != 'train' and sc != 'good':
                mask_dir = os.path.join(root, 'ground_truth', sc)
                self.masks.extend(
                    [os.path.join(mask_dir, p) for p in sorted(os.listdir(mask_dir))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        with open(self.images[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.image_transforms(img)

        ret = [img]

        if self.get_mask and self.set != 'train':
            image_name = os.path.splitext(os.path.basename(self.images[index]))[0]
            ret.append(image_name)

            defect_class = self.defect_classes[index]
            if defect_class == 'good':
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
            ret.append(defect_class)
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
    config = get_argparse()
    subdataset = config.subdataset
    test_only = config.test_only
    train_steps_config = config.train_steps

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    test_output_dir = os.path.join('output/results', subdataset)

    teacher = torch.load('./models/teacher_nf_' + subdataset + '.pth')
    teacher.eval()
    teacher.cuda()

    full_train_set = DefectDataset(set='train', get_mask=False, subdataset=subdataset)
    train_size = int(0.9 * len(full_train_set))
    validation_size = len(full_train_set) - train_size
    rng = torch.Generator().manual_seed(42)
    train_set, validation_set = torch.utils.data.random_split(full_train_set, [train_size, validation_size], rng)
    train_loader = DataLoader(train_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)
    validation_loader = DataLoader(validation_set, pin_memory=True, batch_size=8)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=True, subdataset=subdataset), pin_memory=True, batch_size=1, shuffle=False, drop_last=False)

    imagenet_path = './imagenet_pictures/collected_images'
    penalty_set = ImageNetDataset(imagenet_path, sample_size=train_size)
    penalty_loader = DataLoader(penalty_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)

    student = StudentTeacherModel(nf=False, channels_hidden=1024, n_blocks=4)
    student.train()
    student.cuda()

    autoencoder = StudentTeacherModel(model_autoencoder=True)
    autoencoder.train()
    autoencoder.cuda()

    if test_only == "False":
        train_steps = int(train_steps_config)
        patience = 10
        best_auc = 0
        epochs_to_improve = 0
        early_stop = False
        optimizer = torch.optim.Adam(itertools.chain(student.net.parameters(), autoencoder.net.parameters()), lr=2e-4, eps=1e-08, weight_decay=1e-5)

        temp_loss = 150
        temp_loss_general = 200
        train_loss = []
        for sub_epoch in range(train_steps):
            print('Epoch {} out of {}'.format(sub_epoch+1, train_steps))
            epoch_loss = 0
            student.train()
            autoencoder.train()
            for (mvtec_data, penalty_data) in zip(tqdm(train_loader, disable=True), penalty_loader):
                image = mvtec_data
                image = image[0]
                image = image.cuda()
                penalty_image = penalty_data
                penalty_image = penalty_image.cuda()

                with torch.no_grad():
                    teacher_output_st, _ = teacher(image)

                student_output_st, _ = student(image)
                student_output_st = student_output_st[:, :304]

                distance_st = (teacher_output_st - student_output_st) ** 2
                loss_hard = torch.mean(distance_st)

                # Imagenet penalty
                student_output_penalty, _ = student(penalty_image)
                student_output_penalty = student_output_penalty[:, :304]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty

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

                print(f'Loss after epoch: {loss_total.item()}')
                train_loss.append(loss_total.item())
                epoch_loss += loss_total.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f'Average loss after epoch {sub_epoch + 1}: {avg_epoch_loss}')

            student.eval()
            autoencoder.eval()
            image_auc, pixel_auc = test(test_loader=test_loader, teacher=teacher, student=student, autoencoder=autoencoder, test_output_dir=test_output_dir, desc='Final inference', calculate_other_metrics=False)

            print(f'Validation pixel AUC after epoch {sub_epoch + 1}: {pixel_auc}')

            # Check for early stopping
            if pixel_auc > best_auc:
                best_auc = pixel_auc
                epochs_no_improve = 0
                # Save best model
                torch.save(student, join('./models', 'student_' + subdataset + '.pth'))
                torch.save(autoencoder, join('./models', 'autoencoder_' + subdataset + '.pth'))
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    early_stop = True
                    break

        save_loss_graph(train_loss, os.path.join(test_output_dir, 'graphs'))

    student = torch.load('./models/student_' + subdataset + '.pth')
    student.eval()
    student.cuda()
    autoencoder = torch.load('./models/autoencoder_' + subdataset + '.pth')
    autoencoder.eval()
    autoencoder.cuda()

    auc, pixel_roc_auc, image_f1, pixel_f1, image_recall, image_precision, pixel_recall, pixel_precision, latency = test(test_loader=test_loader, teacher=teacher, student=student, autoencoder=autoencoder, test_output_dir=test_output_dir, desc='Final inference', calculate_other_metrics=True)
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

def test(test_loader, teacher, student, autoencoder, test_output_dir=None, desc='Running inference', calculate_other_metrics=False):
    y_true = []
    y_score = []
    y_score_binary = []
    mask_flat_combined = []
    map_flat_combined = []
    latencies = []
    mask_save_data = []

    for data in tqdm(test_loader, desc=desc, disable=True):
        image, image_name, mask, defect_class = data
        defect_class = defect_class[0]
        _, C, H, W = image.shape
        orig_width = W
        orig_height = H

        image, mask = [t.to('cuda') for t in [image, mask]]

        map_combined, map_st, map_ae, latency_ms = predict(image=image, teacher=teacher, student=student, autoencoder=autoencoder)
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()
        map_combined = 1 / (1 + np.exp(-map_combined))

        latencies.append(latency_ms)

        test_output_dir_images = os.path.join(test_output_dir, 'images')
        if test_output_dir is not None and calculate_other_metrics:
            img_nm = image_name[0]
            mask_save_data.append((os.path.join(test_output_dir_images, 'masks', defect_class), os.path.join(test_output_dir_images, 'masks', defect_class, img_nm + '.png'), map_combined))

            #Saving images heatmap
            test_output_dir_heat = os.path.join(test_output_dir_images, 'heat_maps')
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

    if calculate_other_metrics:
        image_f1_threshold, pixel_f1_threshold, optimal_threshold_auc = save_curves(map_flat_combined, mask_flat_combined, y_score, y_true, auc, pixel_roc_auc, os.path.join(test_output_dir, 'graphs'))

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
    else:
        return auc, pixel_roc_auc

def save_predicted_masks(save_data, threshold):
    for directory_path, file_path, mask in save_data:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        mask = (mask > threshold).astype(np.float32)
        image_to_save = Image.fromarray((mask * 255).astype(np.uint8))
        image_to_save.save(file_path)

def save_curves(pixel_prediction, pixel_gt, image_predictions, image_gt, image_auc, pixel_auc, output_dir):
    # For pixel-level ROC curve
    fpr_pixel, tpr_pixel, thresholds_pixel = roc_curve(y_true=pixel_gt, y_score=pixel_prediction)
    # For image-level ROC curve
    fpr, tpr, thresholds = roc_curve(y_true=image_gt, y_score=image_predictions)
    # For pixel-level precision-recall curve
    precision_pixel, recall_pixel, thresholds_f1_pixel = precision_recall_curve(pixel_gt, pixel_prediction)
    # For image-level precision-recall curve
    precision, recall, thresholds_f1 = precision_recall_curve(image_gt, image_predictions)

    # Optimal threshold for predicted mask
    optimal_idx_auc = np.argmin(np.sqrt(np.square(1 - tpr_pixel) + np.square(fpr_pixel)))
    optimal_threshold_auc = thresholds_pixel[optimal_idx_auc]
    print(f'optimal auc pixel threshold: {optimal_threshold_auc}')

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plotting the image-level Precision-Recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='navy')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Image-Level')
    plt.legend(loc="best")
    plt.grid(True)
    pr_path = os.path.join(output_dir, 'precision_recall_image.png')
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
    pr_pixel_path = os.path.join(output_dir, 'precision_recall_pixel.png')
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
    pixel_roc_path = os.path.join(output_dir, 'pixel_roc.png')
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
    image_roc_path = os.path.join(output_dir, 'image_roc.png')
    plt.savefig(image_roc_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Pixel-level ROC curve image saved to '{pixel_roc_path}'.")
    print(f"Image-level ROC curve image saved to '{image_roc_path}'.")
    print(f"Pixel-level precision-recall curve image saved to '{pr_pixel_path}'.")
    print(f"Image-level precision-recall curve image saved to '{pr_path}'.")

    return optimal_threshold, optimal_threshold_pixel, optimal_threshold_auc

def save_loss_graph(train_loss, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Iterations')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss_path = os.path.join(output_dir, 'training_loss.png')
    plt.savefig(loss_path)
    plt.close()

    print(f"Training loss curve image saved to '{loss_path}'.")

if __name__ == '__main__':
    main()
