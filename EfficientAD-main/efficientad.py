#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import itertools
import random
import time
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
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

random.seed(42)

def get_argparse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--subdataset', default='bottle')
    parser.add_argument('-t', '--test_only', default=False)
    parser.add_argument('-v', '--train_steps', default=100)
    parser.add_argument('-d', '--dataset_dir', default='mvtec_anomaly_detection')
    parser.add_argument('-i', '--imagenet_dir', default='imagenet_pictures/collected_images')
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
    def __init__(self, set='train', get_mask=False, subdataset='bottle', dataset_dir='mvtec'):
        super(DefectDataset, self).__init__()
        self.set = set
        self.masks = list()
        self.images = list()
        self.class_names = ['good']
        self.get_mask = get_mask
        self.defect_classes = []
        self.image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
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
                self.defect_classes.append(sc)

            if self.get_mask and self.set != 'train':
                if sc == 'good':
                    self.masks.extend('mask_is_good' for i in img_paths)
                else:
                    mask_dir = os.path.join(root, 'ground_truth', sc)
                    if (dataset_dir == 'mvtec_loco_anomaly_detection'):
                        mask_sub_dirs = [os.path.join(mask_dir, os.path.splitext(p)[0]) for p in img_paths]
                        self.masks.extend([os.path.join(sub_dir, "000.png") for sub_dir in mask_sub_dirs])
                    else:
                        self.masks.extend(
                        [os.path.join(mask_dir, p) for p in sorted(os.listdir(mask_dir)) if p.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp'))])


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

            if self.masks[index] == 'mask_is_good':
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

            defect_class = self.defect_classes[index]
            ret.append(defect_class)
        return ret

def main():
    start_time = time.time()
    config = get_argparse()
    subdataset = config.subdataset
    test_only = config.test_only
    train_steps = int(config.train_steps)
    dataset_dir = config.dataset_dir
    imagenet_dir = config.imagenet_dir

    test_output_dir = os.path.join('output/results', subdataset)

    teacher = torch.load('./models/teacher_nf_' + subdataset + '.pth', map_location=torch.device(device))
    teacher.eval()
    teacher.to(device)

    full_train_set = DefectDataset(set='train', get_mask=False, subdataset=subdataset, dataset_dir=dataset_dir)
    train_size = int(len(full_train_set))
    train_loader = DataLoader(full_train_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)
    test_loader = DataLoader(DefectDataset(set='test', get_mask=True, subdataset=subdataset, dataset_dir=dataset_dir), pin_memory=True, batch_size=1, shuffle=False, drop_last=False)

    penalty_set = ImageNetDataset(imagenet_dir, sample_size=train_size)
    penalty_loader = DataLoader(penalty_set, pin_memory=True, batch_size=8, shuffle=True, drop_last=True)

    student = StudentTeacherModel(nf=False, channels_hidden=1024, n_blocks=4)
    student.train()
    student.to(device)

    autoencoder = StudentTeacherModel(model_autoencoder=True)
    autoencoder.train()
    autoencoder.to(device)

    if test_only == "False":
        final_training_epoch = None
        first_loss = None
        last_loss = None
        patience = 10
        best_pixel_auc = 0
        best_image_auc = 0
        early_stop = False
        optimizer = torch.optim.Adam(itertools.chain(student.net.parameters(), autoencoder.net.parameters()), lr=2e-4, eps=1e-08, weight_decay=1e-5)

        train_loss = []
        for sub_epoch in range(train_steps):
            print('Epoch {} out of {}'.format(sub_epoch+1, train_steps))
            epoch_loss = 0
            student.train()
            autoencoder.train()
            for (mvtec_data, penalty_data) in zip(tqdm(train_loader, disable=True), penalty_loader):
                image = mvtec_data
                image = image[0]
                image = image.to(device)
                penalty_image = penalty_data
                penalty_image = penalty_image.to(device)

                with torch.no_grad():
                    teacher_output, _ = teacher(image)

                student_output, _ = student(image)
                student_output_st = student_output[:, :304]

                # Compare with teacher output
                distance_st = (teacher_output - student_output_st) ** 2
                loss_hard = torch.mean(distance_st)

                # Imagenet penalty
                student_output_penalty, _ = student(penalty_image)
                student_output_penalty = student_output_penalty[:, :304]
                loss_penalty = torch.mean(student_output_penalty**2)
                loss_st = loss_hard + loss_penalty

                # Compare with ae output
                ae_output = autoencoder(image)
                student_output_ae = student_output[:, 304:]
                distance_ae = (teacher_output - ae_output) ** 2
                distance_stae = (ae_output - student_output_ae) ** 2
                loss_ae = torch.mean(distance_ae)
                loss_stae = torch.mean(distance_stae)
                loss_total = loss_st + loss_ae + loss_stae

                optimizer.zero_grad()
                loss_total.backward()
                optimizer.step()

                epoch_loss += loss_total.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            if sub_epoch == 0:
                first_loss = avg_epoch_loss
            train_loss.append(avg_epoch_loss)
            print(f'Average loss after epoch {sub_epoch + 1}: {avg_epoch_loss}')

            student.eval()
            autoencoder.eval()
            image_auc, pixel_auc = test(test_loader=test_loader, teacher=teacher, student=student, autoencoder=autoencoder, test_output_dir=test_output_dir, desc='Final inference', calculate_other_metrics=False)

            print(f'Validation pixel AUC after epoch {sub_epoch + 1}: {pixel_auc}')
            print(f'Validation image AUC after epoch {sub_epoch + 1}: {image_auc}')

            # Check for early stopping
            if pixel_auc > best_pixel_auc or image_auc > best_image_auc:
                if pixel_auc > best_pixel_auc:
                    best_pixel_auc = pixel_auc
                if image_auc > best_image_auc:
                    best_image_auc = image_auc
                epochs_no_improve = 0

                torch.save(student, join('./models', 'student_' + subdataset + '.pth'))
                torch.save(autoencoder, join('./models', 'autoencoder_' + subdataset + '.pth'))
                final_training_epoch = sub_epoch + 1
                last_loss = avg_epoch_loss
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print('Early stopping!')
                    early_stop = True
                    break

        save_loss_graph(train_loss, os.path.join(test_output_dir, 'graphs'), 'efficientad_training_loss.png')

        if not early_stop:
            last_loss = train_loss[-1]
            final_training_epoch = train_steps

        training_time = time.time() - start_time
        print(f'Time that it took for training: {training_time}')

    student = torch.load('./models/student_' + subdataset + '.pth', map_location=torch.device(device))
    student.eval()
    student.to(device)
    autoencoder = torch.load('./models/autoencoder_' + subdataset + '.pth', map_location=torch.device(device))
    autoencoder.eval()
    autoencoder.to(device)

    auc, pixel_roc_auc, image_f1, pixel_f1, image_recall, image_precision, pixel_recall, pixel_precision, latency = test(test_loader=test_loader, teacher=teacher, student=student, autoencoder=autoencoder, test_output_dir=test_output_dir, desc='Final inference', calculate_other_metrics=True)

    with open(os.path.join('./output', 'results', subdataset, 'efficientad_metrics.txt'), 'w') as file:
        file.write('Final pixel auc: {:.4f}'.format(pixel_roc_auc))
        file.write('\nFinal image auc: {:.4f}'.format(auc))
        file.write('\nFinal pixel f1: {:.4f}'.format(pixel_f1))
        file.write('\nFinal image f1: {:.4f}'.format(image_f1))
        file.write('\nFinal pixel recall: {:.4f}'.format(pixel_recall))
        file.write('\nFinal image recall: {:.4f}'.format(image_recall))
        file.write('\nFinal pixel precision: {:.4f}'.format(pixel_precision))
        file.write('\nFinal image precision: {:.4f}'.format(image_precision))
        file.write('\nFinal average processing latency: {:.4f} ms/img'.format(latency))
        if test_only == "False":
            file.write('\nFinal training time: {:.4f}'.format(training_time))
            file.write('\nFinal epoch ' + str(final_training_epoch))
            file.write('\nLoss: {:.4f} -> {:.4f}'.format(first_loss, last_loss))

def test(test_loader, teacher, student, autoencoder, test_output_dir=None, desc='Running inference', calculate_other_metrics=False):
    y_true = []
    y_score = []
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

        image, mask = [t.to(device) for t in [image, mask]]

        map_combined, map_st, map_ae, latency_ms = predict(image=image, teacher=teacher, student=student, autoencoder=autoencoder)
        map_combined = torch.nn.functional.interpolate(map_combined, (orig_height, orig_width), mode='bilinear')
        map_combined = map_combined[0, 0].cpu().numpy()

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

    # pixel-level auroc calculations
    pixel_roc_auc = roc_auc_score(y_true=mask_flat_combined, y_score=map_flat_combined)
    pixel_roc_auc = pixel_roc_auc * 100
    # image-level auroc calculations
    auc = roc_auc_score(y_true=y_true, y_score=y_score)
    auc = auc * 100

    if calculate_other_metrics:
        image_f1_threshold, pixel_f1_threshold = save_curves(map_flat_combined, mask_flat_combined, y_score, y_true, auc, pixel_roc_auc, os.path.join(test_output_dir, 'graphs'))

        # Saving predicted masks as png with the best calculated threshold:
        save_predicted_masks(mask_save_data, pixel_f1_threshold)

        # Convert image and pixel predictions to binary with calculated thresholds
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

@torch.no_grad()
def predict(image, teacher, student, autoencoder):
    start_time = time.time()

    teacher_output, _ = teacher(image)
    student_output, _ = student(image)
    autoencoder_output = autoencoder(image)

    end_time = time.time()
    latency_ms = (end_time - start_time) * 1000

    # Calculate the mean squared error (MSE) between the teacher and student outputs
    map_st = torch.mean((teacher_output - student_output[:, :304]) ** 2, dim=1, keepdim=True)
    # Calculate the MSE between the autoencoder output and the student output
    map_ae = torch.mean((autoencoder_output - student_output[:, 304:]) ** 2, dim=1, keepdim=True)
    # Combine prediction maps of student and autoencoder
    map_combined = 0.5 * map_st + 0.5 * map_ae
    return map_combined, map_st, map_ae, latency_ms

if __name__ == '__main__':
    main()
