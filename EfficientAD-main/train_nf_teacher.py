#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import time
from os.path import join

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import *

if torch.cuda.is_available():
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
else:
    device = "cpu"

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdataset', default='bottle')
    parser.add_argument('-v', '--train_steps', default=240)
    parser.add_argument('-d', '--dataset_dir', default='mvtec_anomaly_detection')
    return parser.parse_args()

class DefectDataset(Dataset):
    def __init__(self, set='train', subdataset='bottle', dataset_dir='mvtec'):
        super(DefectDataset, self).__init__()
        self.set = set
        self.labels = list()
        self.images = list()
        self.class_names = ['good']
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        root = join(dataset_dir, subdataset)
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

        self.features = np.load(os.path.join('data', 'features', subdataset, set + '.npy'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        fg = torch.ones([1, 192, 192])

        with open(self.images[index], 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.image_transforms(img)

        label = self.labels[index]
        feat = self.features[index]

        ret = [fg, label, img, feat]
        return ret

def train(train_loader, test_loader, subdataset='bottle', train_steps=240):
    start_time = time.time()

    teacher = StudentTeacherModel(nf=True)
    teacher.to(device)
    optimizer = torch.optim.Adam(teacher.net.parameters(), lr=2e-4, eps=1e-08, weight_decay=1e-5)
    mean_nll_obs = Score_Observer('AUROC mean over maps')
    max_nll_obs = Score_Observer('AUROC  max over maps')

    train_loss_full = []
    first_loss = None

    for epoch in range(train_steps):
        teacher.train()
        print(F'\nTrain epoch {epoch}')
        train_loss = list()
        for i, data in enumerate(tqdm(train_loader, disable=True)):
            optimizer.zero_grad()

            fg, labels, image, features = data
            fg, labels, image, features = [t.to(device) for t in [fg, labels, image, features]]

            # Downsample foreground mask to match the model output size
            fg_down = downsampling(fg, (24, 24), bin=False)

            z, jac = teacher(features, extract_features=False)

            loss = get_nf_loss(z, jac, fg_down)
            train_loss.append(t2np(loss))

            loss.backward()
            optimizer.step()

        # Calculate mean training loss for the epoch
        mean_train_loss = np.mean(train_loss)
        train_loss_full.append(mean_train_loss)
        print('Epoch: {:d} \t teacher train loss: {:.4f}'.format(epoch, mean_train_loss))

        # Peform intermediate evaluation
        if epoch % 24 == 0:
            teacher.eval()
            test_loss = list()
            test_labels = list()
            img_nll = list()
            max_nlls = list()

            with torch.no_grad():
                for i, data in enumerate(tqdm(test_loader, disable=True)):
                    fg, labels, image, features = data
                    fg, image, features = [t.to(device) for t in [fg, image, features]]

                    fg_down = downsampling(fg, (24, 24), bin=False)
                    z, jac = teacher(features, extract_features=False)
                    # Calculate loss per-sample
                    loss = get_nf_loss(z, jac, fg_down, per_sample=True)
                    # Calculate loss per-pixel
                    nll = get_nf_loss(z, jac, fg_down, per_pixel=True)

                    img_nll.append(t2np(loss))
                    # Track max loss over all pixels
                    max_nlls.append(np.max(t2np(nll), axis=(-1, -2)))
                    # Calculate mean test loss
                    test_loss.append(loss.mean().item())
                    test_labels.append(labels)

            img_nll = np.concatenate(img_nll)
            max_nlls = np.concatenate(max_nlls)
            test_loss = np.mean(np.array(test_loss))

            print('Epoch: {:d} \t teacher test_loss: {:.4f}'.format(epoch, test_loss))

            test_labels = np.concatenate(test_labels)
            # Prepare anomaly labels
            is_anomaly = np.array([0 if l == 0 else 1 for l in test_labels])

            # Calculate auroc per-sample
            mean_nll_obs.update(roc_auc_score(is_anomaly, img_nll), epoch,
                                print_score='True' or epoch == 3 - 1)
            # Calculate auroc per-pixel
            max_nll_obs.update(roc_auc_score(is_anomaly, max_nlls), epoch,
                               print_score='True' or epoch == 3 - 1)
        if epoch == 0:
            first_loss = mean_train_loss

    last_loss = train_loss_full[-1]
    save_loss_graph(train_loss_full, os.path.join('./output', 'results', subdataset, 'graphs'), 'teacher_training_loss.png')
    train_time = time.time() - start_time

    teacher.to('cpu')
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(teacher, join('./models', 'teacher_nf_' + subdataset + '.pth'))
    print('teacher saved!')
    teacher.to(device)

    with open(os.path.join('./output', 'results', subdataset, 'teacher_metrics.txt'), 'w') as file:
        file.write('Final auc mean: {:.4f}\n'.format(np.mean(mean_nll_obs.last_score)))
        file.write('Final auc max: {:.4f}\n'.format(np.mean(max_nll_obs.last_score)))
        file.write('\nLoss: {:.4f} -> {:.4f}'.format(first_loss, last_loss))
        file.write('\nTraining time: {:.4f}\n'.format(train_time))

def main_teacher():
    config = get_argparse()
    subdataset = config.subdataset
    train_steps = int(config.train_steps)
    dataset_dir = config.dataset_dir

    print('\nTrain class ' + subdataset)

    train_loader = DataLoader(DefectDataset(set='train', subdataset=subdataset, dataset_dir=dataset_dir), pin_memory=True,
                              batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(DefectDataset(set='test', subdataset=subdataset, dataset_dir=dataset_dir), pin_memory=True,
                             batch_size=16, shuffle=False, drop_last=False)

    train(train_loader, test_loader, subdataset, train_steps)

if __name__ == '__main__':
    main_teacher()
