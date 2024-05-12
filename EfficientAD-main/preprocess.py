import argparse
from os.path import join

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
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
    parser.add_argument('-d', '--dataset_dir', default='mvtec_anomaly_detection')
    return parser.parse_args()

def make_dataloaders(trainset, testset, shuffle_train=True, drop_last=True):
    trainloader = torch.utils.data.DataLoader(trainset, pin_memory=True, batch_size=8, shuffle=shuffle_train,
                                              drop_last=drop_last)
    testloader = torch.utils.data.DataLoader(testset, pin_memory=True, batch_size=16, shuffle=False,
                                             drop_last=False)
    return trainloader, testloader

def load_img_datasets(dataset_dir, class_name):
    '''
    Expected folder/file format to find anomalies of class <class_name> from dataset location <dataset_dir>:

    If 3D data is available (as for MVTec3D):

        train data:

            RGB data:
                dataset_dir/class_name/train/good/rgb/any_filename.png
                dataset_dir/class_name/train/good/rgb/another_filename.tif
                [...]

            3D data:
                dataset_dir/class_name/train/good/xyz/abc123.tiff
                dataset_dir/class_name/train/good/xyz/def1337.tiff
                [...]

        test data:


            'normal data' = non-anomalies

                see 'train data' and replace 'train' with 'test'

            anomalies - assume there is an anomaly classes 'crack'

                RGB data:
                    dataset_dir/class_name/test/crack/rgb/dat_crack_damn.png
                    dataset_dir/class_name/test/crack/rgb/let_it_crack.png
                    dataset_dir/class_name/test/crack/rgb/writing_docs_is_fun.png
                    [...]

                3D data:
                    dataset_dir/class_name/test/curved/xyz/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
                    dataset_dir/class_name/test/curved/xyz/but_this_code_is_practicable_for_the_mvtec_dataset.png
                    [...]

    else:

        train data:

                dataset_dir/class_name/train/good/any_filename.png
                dataset_dir/class_name/train/good/another_filename.tif
                dataset_dir/class_name/train/good/xyz.png
                [...]

        test data:

            'normal data' = non-anomalies

                dataset_dir/class_name/test/good/name_the_file_as_you_like_as_long_as_there_is_an_image_extension.webp
                dataset_dir/class_name/test/good/did_you_know_the_image_extension_webp?.png
                dataset_dir/class_name/test/good/did_you_know_that_filenames_may_contain_question_marks????.png
                dataset_dir/class_name/test/good/dont_know_how_it_is_with_windows.png
                dataset_dir/class_name/test/good/just_dont_use_windows_for_this.png
                [...]

            anomalies - assume there are anomaly classes 'crack' and 'curved'

                dataset_dir/class_name/test/crack/dat_crack_damn.png
                dataset_dir/class_name/test/crack/let_it_crack.png
                dataset_dir/class_name/test/crack/writing_docs_is_fun.png
                [...]

                dataset_dir/class_name/test/curved/wont_make_a_difference_if_you_put_all_anomalies_in_one_class.png
                dataset_dir/class_name/test/curved/but_this_code_is_practicable_for_the_mvtec_dataset.png
                [...]
    '''

    def target_transform(target):
        return class_perm[target]

    data_dir_train = os.path.join(dataset_dir, class_name, 'train')
    data_dir_test = os.path.join(dataset_dir, class_name, 'test')
    classes = os.listdir(data_dir_test)
    if 'good' not in classes:
        raise RuntimeError(
            'There should exist a subdirectory "good". Read the doc of this function for further information.')
    classes.sort()
    class_perm = list()
    class_idx = 1
    for cl in classes:
        if cl == 'good':
            class_perm.append(0)
        else:
            class_perm.append(class_idx)
            class_idx += 1

    image_transforms = transforms.Compose([transforms.Resize((768, 768)), transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    valid_img = None
    trainset = ImageFolder(data_dir_train, transform=image_transforms, is_valid_file=valid_img)
    testset = ImageFolder(data_dir_test, transform=image_transforms, target_transform=target_transform,
                          is_valid_file=valid_img)
    return trainset, testset

def extract_image_features(base_dir, extract_layer=35):
    model = FeatureExtractor(layer_idx=extract_layer)
    model.to(device)
    model.eval()

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(join(base_dir, d))]

    for class_name in classes:
        print(class_name)
        train_set, test_set = load_img_datasets(base_dir, class_name)
        train_loader, test_loader = make_dataloaders(train_set, test_set, shuffle_train=False, drop_last=False)
        for name, loader in zip(['train', 'test'], [train_loader, test_loader]):
            features = list()
            for i, data in enumerate(tqdm(loader)):
                img = data[0].to(device)
                with torch.no_grad():
                    z = model(img)
                features.append(t2np(z))

            features = np.concatenate(features, axis=0)
            export_dir = join('data', 'features', class_name)
            os.makedirs(export_dir, exist_ok=True)
            print(export_dir)
            np.save(join(export_dir, f'{name}.npy'), features)

extract_image_features(get_argparse().dataset_dir)