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
from sklearn.metrics import roc_auc_score
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F
from freia_funcs import *
import torch.nn as nn
from PIL import Image
from os.path import join
from scipy.ndimage.morphology import binary_dilation
from torchvision.datasets import ImageFolder
from efficientad import StudentTeacherModel, get_nf, FeatureExtractor
import matplotlib.pyplot as plt
import time

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdataset', default='bottle')
    return parser.parse_args()

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

class DefectDataset(Dataset):
    def __init__(self, set='train', get_mask=True, subdataset='bottle'):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
                self.labels.append(label)


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

    def __getitem__(self, index):
        fg = torch.ones([1, 192, 192])

        with open(self.images[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.image_transforms(img)

        label = self.labels[index]

        ret = [fg, label, img]
        return ret

def train(train_loader, test_loader, subdataset='bottle'):
    start_time = time.time()
    teacher = StudentTeacherModel(nf=True)
    teacher.cuda()
    optimizer = torch.optim.Adam(teacher.net.parameters(), lr=2e-4, eps=1e-08, weight_decay=1e-5)
    # Observers to track AUROC scores during training.
    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    patience = 24
    best_mean_auc = 0
    best_max_auc = 0
    epochs_to_improve = 0
    early_stop = False
    train_loss_full = []

    first_loss = None
    last_loss = None
    final_training_epoch = None

    for epoch in range(240):
        teacher.train()
        print(F'\nTrain epoch {epoch}')
        train_loss = list()
        for i, data in enumerate(tqdm(train_loader, disable=True)):
            # Clear gradients.
            optimizer.zero_grad()

            # Unpack data and move to device.
            fg, labels, image = data
            fg, labels, image = [t.to('cuda') for t in [fg, labels, image]]

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
        train_loss_full.append(mean_train_loss)
        print('Epoch: {:d} \t teacher train loss: {:.4f}'.format(epoch, mean_train_loss))

        teacher.eval()
        test_loss = list()
        test_labels = list()
        img_nll = list()
        max_nlls = list()

        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, disable=True)):
                # Unpack and move data to device, similar to training phase.
                fg, labels, image = data
                fg, image = [t.to('cuda') for t in [fg, image]]

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

        print('Epoch: {:d} \t teacher test_loss: {:.4f}'.format(epoch, test_loss))

        test_labels = np.concatenate(test_labels)
        # Prepare anomaly labels.
        is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

        mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                            print_score='True' or epoch == 3 - 1)
        max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                           print_score='True' or epoch == 3 - 1)

        epoch_mean_auc = np.mean(mean_nll_obs.last_score)
        epoch_max_auc = np.mean(max_nll_obs.last_score)

        if epoch_mean_auc > best_mean_auc or epoch_max_auc > best_max_auc:
            if epoch_mean_auc > best_mean_auc:
                best_mean_auc = epoch_mean_auc
            if epoch_max_auc > best_max_auc:
                best_max_auc = epoch_max_auc
            epochs_no_improve = 0
            # Save best model
            teacher.to('cpu')
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(teacher, join('./models', 'teacher_nf_' + subdataset + '.pth'))
            print('teacher saved!')
            teacher.to('cuda')
            last_loss = mean_train_loss
            final_training_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve == patience:
                print('Early stopping!')
                early_stop = True
                break

        if epoch == 0:
            first_loss = mean_train_loss

    if not early_stop:
        last_loss = train_loss_full[-1]
        final_training_epoch = 240

    save_loss_graph(train_loss_full, os.path.join('./output', 'results', subdataset, 'graphs'))

    train_time = time.time() - start_time

    with open(os.path.join('./output', 'results', subdataset, 'teacher_metrics.txt'), 'w') as file:
        file.write('Final auc mean: {:.4f}\n'.format(best_mean_auc))
        file.write('Final auc max: {:.4f}\n'.format(best_max_auc))
        file.write('Early stop: ' + str(early_stop))
        file.write('\nLoss: {:.4f} -> {:.4f}'.format(first_loss, last_loss))
        file.write('\nTraining time: {:.4f}\n'.format(train_time))
        file.write('Final training epoch ' + str(final_training_epoch))

    return teacher, mean_nll_obs, max_nll_obs

def main_teacher():
    config = get_argparse()
    subdataset = config.subdataset

    max_scores = list()
    mean_scores = list()

    print('\nTrain class ' + subdataset)
    train_loader = DataLoader(DefectDataset(set='train', get_mask=False, subdataset=subdataset), pin_memory=True,
                              batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=False, subdataset=subdataset), pin_memory=True,
                             batch_size=16, shuffle=False, drop_last=False)
    teacher, mean_sc, max_sc = train(train_loader, test_loader, subdataset)
    mean_scores.append(mean_sc)
    max_scores.append(max_sc)

    last_mean = np.mean([s.last_score for s in mean_scores])
    last_max = np.mean([s.last_score for s in max_scores])
    best_mean = np.mean([s.best_score for s in mean_scores])
    best_max = np.mean([s.best_score for s in max_scores])
    print('\nAUROC % after last epoch\n\tmean over maps: {:.2f} \t max over maps: {:.2f}'.format(last_mean, last_max))
    print('best AUROC %\n\tmean over maps: {:.2f} \t max over maps: {:.2f}'.format(best_mean, best_max))

    return teacher

def save_loss_graph(train_loss, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    loss_path = os.path.join(output_dir, 'teacher_training_loss.png')
    plt.savefig(loss_path)
    plt.close()

    print(f"Training loss curve image saved to '{loss_path}'.")

if __name__ == '__main__':
    main_teacher()