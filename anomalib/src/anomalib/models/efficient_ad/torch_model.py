"""Torch model for student, teacher and autoencoder model in EfficientAd"""

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from enum import Enum
from torch import Tensor, nn
from torchvision import transforms
from anomalib.models.efficient_ad.freia_funcs import *

logger = logging.getLogger(__name__)

# Normalize images according to ImageNet standards
def imagenet_norm_batch(x):
    # Predefined mean and std for normalization
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(x.device)
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(x.device)
    x_norm = (x - mean) / std # Normalize
    return x_norm

# Reduce the number of elements in a tensor for efficient processing
def reduce_tensor_elems(tensor: torch.Tensor, m=2**24) -> torch.Tensor:
    """Flattens n-dimensional tensors,  selects m elements from it
    and returns the selected elements as tensor. It is used to select
    at most 2**24 for torch.quantile operation, as it is the maximum
    supported number of elements.
    https://github.com/pytorch/pytorch/blob/b9f81a483a7879cd3709fd26bcec5f1ee33577e6/aten/src/ATen/native/Sorting.cpp#L291

    Args:
        tensor (torch.Tensor): input tensor from which elements are selected
        m (int): number of maximum tensor elements. Default: 2**24

    Returns:
            Tensor: reduced tensor
    """
    tensor = torch.flatten(tensor)
    # If tensor is too big, sample a subset
    if len(tensor) > m:
        # select a random subset with m elements.
        perm = torch.randperm(len(tensor), device=tensor.device)
        idx = perm[:m]
        tensor = tensor[idx]
    return tensor

# Enum for model sizes (small, medium)
class EfficientAdModelSize(str, Enum):
    """Supported EfficientAd model sizes"""

    M = "medium"
    S = "small"

# Small Patch Description Network (PDN)
#TODO: Use student model of AST
class PDN_S(nn.Module):
    """Patch Description Network small

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        # Define convolutional layers with optional padding
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 128, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv4 = nn.Conv2d(256, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        # Define average pooling layers with optional padding
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    # Process input through layers and apply activations
    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x

class PDN_M(nn.Module):
    """Patch Description Network medium

    Args:
        out_channels (int): number of convolution output channels
    """

    def __init__(self, out_channels: int, padding: bool = False) -> None:
        super().__init__()
        pad_mult = 1 if padding else 0
        self.conv1 = nn.Conv2d(3, 256, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=3 * pad_mult)
        self.conv3 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1 * pad_mult)
        self.conv5 = nn.Conv2d(512, out_channels, kernel_size=4, stride=1, padding=0 * pad_mult)
        self.conv6 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0 * pad_mult)
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=1 * pad_mult)

    def forward(self, x):
        x = imagenet_norm_batch(x)
        x = F.relu(self.conv1(x))
        x = self.avgpool1(x)
        x = F.relu(self.conv2(x))
        x = self.avgpool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)
        return x

# Encoder part of the Autoencoder
class Encoder(nn.Module):
    """Autoencoder Encoder model."""

    def __init__(self) -> None:
        super().__init__()
        # Define encoding layers to compress input images
        self.enconv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enconv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1)
        self.enconv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.enconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.enconv6 = nn.Conv2d(64, 64, kernel_size=8, stride=1, padding=0)

    def forward(self, x):
        # Encode input image to a smaller representation
        x = F.relu(self.enconv1(x))
        x = F.relu(self.enconv2(x))
        x = F.relu(self.enconv3(x))
        x = F.relu(self.enconv4(x))
        x = F.relu(self.enconv5(x))
        x = self.enconv6(x)
        return x

# Decoder part of the Autoencoder
class Decoder(nn.Module):
    """Autoencoder Decoder model.

    Args:
        out_channels (int): number of convolution output channels
        img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.img_size = img_size
        self.last_upsample = (
            int(img_size[0] / 4) if padding else int(img_size[0] / 4) - 8,
            int(img_size[1] / 4) if padding else int(img_size[1] / 4) - 8,
        )
         # Define decoding layers to expand input images
        self.deconv1 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv2 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv3 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv4 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv5 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv6 = nn.Conv2d(64, 64, kernel_size=4, stride=1, padding=2)
        self.deconv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.deconv8 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.dropout4 = nn.Dropout(p=0.2)
        self.dropout5 = nn.Dropout(p=0.2)
        self.dropout6 = nn.Dropout(p=0.2)

    def forward(self, x):
        # Decode input image to a normal size representation
        x = F.interpolate(x, size=(int(self.img_size[0] / 64) - 1, int(self.img_size[1] / 64) - 1), mode="bilinear")
        x = F.relu(self.deconv1(x))
        x = self.dropout1(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 32), int(self.img_size[1] / 32)), mode="bilinear")
        x = F.relu(self.deconv2(x))
        x = self.dropout2(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 16) - 1, int(self.img_size[1] / 16) - 1), mode="bilinear")
        x = F.relu(self.deconv3(x))
        x = self.dropout3(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 8), int(self.img_size[1] / 8)), mode="bilinear")
        x = F.relu(self.deconv4(x))
        x = self.dropout4(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 4) - 1, int(self.img_size[1] / 4) - 1), mode="bilinear")
        x = F.relu(self.deconv5(x))
        x = self.dropout5(x)
        x = F.interpolate(x, size=(int(self.img_size[0] / 2) - 1, int(self.img_size[1] / 2) - 1), mode="bilinear")
        x = F.relu(self.deconv6(x))
        x = self.dropout6(x)
        x = F.interpolate(x, size=self.last_upsample, mode="bilinear")
        x = F.relu(self.deconv7(x))
        x = self.deconv8(x)
        return x

# AutoEncoder combining Encoder and Decoder
class AutoEncoder(nn.Module):
    """EfficientAd Autoencoder.

    Args:
       out_channels (int): number of convolution output channels
       img_size (tuple): size of input images
    """

    def __init__(self, out_channels, padding, img_size, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = Encoder()
        self.decoder = Decoder(out_channels, padding, img_size)

    def forward(self, x):
        # Encode then decode the input
        x = imagenet_norm_batch(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

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
def get_nf(input_dim=368, channels_hidden=64):
    nodes = list()
    # If positional encoding is enabled, add an input node for it.
    nodes.append(InputNode(32, name='input'))
    # Main input node.
    nodes.append(InputNode(input_dim, name='input'))
    # Creating coupling blocks.
    kernel_sizes = [3, 3, 3, 5]
    for k in range(4):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # Conditional coupling layer if positional encoding is used.
        nodes.append(Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                          {'clamp': 1.9,
                           'F_class': F_conv,
                           'cond_dim': 32,
                           'F_args': {'channels_hidden': channels_hidden,
                                      'kernel_size': kernel_sizes[k]}},
                          name=F'conv_{k}'))
    # Output node.
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    # Creating the reversible graph net.
    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf

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

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()

        self.feature_extractor = FeatureExtractor()
        self.net = get_nf()
        # Positional encoding initialization if enabled.
        self.pos_enc = positionalencoding2d(32, 384, 384)

        # Unshuffle operation for processing depth information.
        self.unshuffle = nn.PixelUnshuffle(8)

    def forward(self, x, depth):
        # Feature extraction based on the mode and configuration.
        with torch.no_grad():
            f = self.feature_extractor(x)

        inp = torch.cat([f, self.unshuffle(depth)], dim=1)

        # Processing through the network with positional encoding.
        cond = self.pos_enc.tile(inp.shape[0], 1, 1, 1)
        z = self.net([cond, inp])

        # Calculating the Jacobian for the normalizing flow.
        jac = self.net.jacobian(run_forward=False)[0]
        # Returning the transformed input and Jacobian.
        return z, jac

class EfficientAdModel(nn.Module):
    """EfficientAd model.

    Args:
        teacher_out_channels (int): number of convolution output channels of the pre-trained teacher model
        pretrained_models_dir (str): path to the pretrained model weights
        input_size (tuple): size of input images
        model_size (str): size of student and teacher model
        padding (bool): use padding in convoluional layers
        pad_maps (bool): relevant if padding is set to False. In this case, pad_maps = True pads the
            output anomaly maps so that their size matches the size in the padding = True case.
        device (str): which device the model should be loaded on
    """
    # Initialize EfficientAd model with configurations for teacher and student networks, and autoencoder
    def __init__(
        self,
        teacher_out_channels: int,
        input_size: tuple[int, int],
        model_size: EfficientAdModelSize = EfficientAdModelSize.S,
        padding=False,
        pad_maps=True,
    ) -> None:
        super().__init__()

        self.pad_maps = pad_maps
        self.teacher: PDN_M | PDN_S
        self.student: PDN_M | PDN_S
        self.teacherv2 = TeacherModel()

        if model_size == EfficientAdModelSize.M:
            self.teacher = PDN_M(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_M(out_channels=teacher_out_channels * 2, padding=padding)

        elif model_size == EfficientAdModelSize.S:
            self.teacher = PDN_S(out_channels=teacher_out_channels, padding=padding).eval()
            self.student = PDN_S(out_channels=teacher_out_channels * 2, padding=padding)

        else:
            raise ValueError(f"Unknown model size {model_size}")

        # Initialize the autoencoder with the specified configuration.
        self.ae: AutoEncoder = AutoEncoder(out_channels=teacher_out_channels, padding=padding, img_size=input_size)
        # Store the number of output channels for the teacher model. This information
        # is essential for setting up the model architecture and for comparison between
        # the teacher and student model outputs.
        self.teacher_out_channels: int = teacher_out_channels
        # Record the expected input image size. This ensures that the model processes
        # inputs correctly and is configured properly for handling images of this size.
        self.input_size: tuple[int, int] = input_size

        # Initialize parameters for normalizing the outputs of the teacher model.
        # The mean and standard deviation are initially set to zeros and will be updated
        # during model training. This normalization helps in stabilizing the model's learning.
        self.mean_std: nn.ParameterDict = nn.ParameterDict(
            {
                "mean": torch.zeros((1, self.teacher_out_channels, 1, 1)),
                "std": torch.zeros((1, self.teacher_out_channels, 1, 1)),
            }
        )

        # Initialize parameters for quantiles used in normalizing anomaly scores.
        # These values are crucial for scaling anomaly maps to a consistent range,
        # making the model's outputs more interpretable and comparable.
        self.quantiles: nn.ParameterDict = nn.ParameterDict(
            {
                "qa_st": torch.tensor(0.0),
                "qb_st": torch.tensor(0.0),
                "qa_ae": torch.tensor(0.0),
                "qb_ae": torch.tensor(0.0),
            }
        )

    # Check if mean and standard deviation parameters are set for normalization
    def is_set(self, p_dic: nn.ParameterDict) -> bool:
        for _, value in p_dic.items():
            if value.sum() != 0:
                return True
        return False

    # Randomly apply image transformations to augment data for robust learning
    def choose_random_aug_image(self, image: Tensor) -> Tensor:
        transform_functions = [
            transforms.functional.adjust_brightness,
            transforms.functional.adjust_contrast,
            transforms.functional.adjust_saturation,
        ]
        # Sample an augmentation coefficient λ from the uniform distribution U(0.8, 1.2)
        coefficient = random.uniform(0.8, 1.2)  # nosec: B311
        transform_function = random.choice(transform_functions)  # nosec: B311
        return transform_function(image, coefficient)

    def forward(self, batch: Tensor, batch_imagenet: Tensor = None, normalize: bool = True) -> Tensor | dict:
        """Prediction by EfficientAd models.

        Args:
            normalize: Normalize anomaly maps or not.
            batch (Tensor): Input images.

        Returns:
            Tensor: Predictions
        """
        with torch.no_grad():
            teacher_output = self.teacher(batch)
            if self.is_set(self.mean_std):
                # Normalize the teacher's output using pre-set mean and std
                teacher_output = (teacher_output - self.mean_std["mean"]) / self.mean_std["std"]
                
        # Generate the student's output from the same input batch for anomaly comparison
        student_output = self.student(batch)
        # Calculate squared difference between teacher and student outputs
        distance_st = torch.pow(teacher_output - student_output[:, : self.teacher_out_channels, :, :], 2)

        if self.training:
            # Calculate the student loss
            # Reduce the dimensionality of the distance tensor to simplify loss calculation
            distance_st = reduce_tensor_elems(distance_st)
            # Find the hardest examples to focus on during training
            d_hard = torch.quantile(distance_st, 0.999)
            # Calculate the loss for these hard examples
            loss_hard = torch.mean(distance_st[distance_st >= d_hard])
            # Calculate penalty for the student's output using augmented ImageNet images
            student_output_penalty = self.student(batch_imagenet)[:, : self.teacher_out_channels, :, :]
            loss_penalty = torch.mean(student_output_penalty**2)
            # Combine the hard loss and penalty for the total student loss
            loss_st = loss_hard + loss_penalty

            # Calculate the autoencoder and combined student-autoencoder loss
            # Augment the input batch for robustness
            aug_img = self.choose_random_aug_image(batch)
             # Get the autoencoder's output for the augmented image
            ae_output_aug = self.ae(aug_img)

            # Normalize teacher's output for augmented image if mean and std are set
            with torch.no_grad():
                teacher_output_aug = self.teacher(aug_img)
                if self.is_set(self.mean_std):
                    teacher_output_aug = (teacher_output_aug - self.mean_std["mean"]) / self.mean_std["std"]
                    
            # Calculate the student's output for the augmented image
            student_output_ae_aug = self.student(aug_img)[:, self.teacher_out_channels :, :, :]

            # Calculate distances for loss between teacher's output and autoencoder, and autoencoder and student
            distance_ae = torch.pow(teacher_output_aug - ae_output_aug, 2)
            distance_stae = torch.pow(ae_output_aug - student_output_ae_aug, 2)

            # Calculate losses for autoencoder and the combination of student and autoencoder
            loss_ae = torch.mean(distance_ae)
            loss_stae = torch.mean(distance_stae)
            # Return the combined losses
            return (loss_st, loss_ae, loss_stae)

        else:
            # In evaluation mode, calculate anomaly maps instead of losses
            with torch.no_grad():
                 # Get the autoencoder output for the original batch
                ae_output = self.ae(batch)

            # Calculate mean squared distance between teacher and student outputs
            map_st = torch.mean(distance_st, dim=1, keepdim=True)
            # Calculate mean squared distance for autoencoder and student outputs
            map_stae = torch.mean(
                (ae_output - student_output[:, self.teacher_out_channels :]) ** 2, dim=1, keepdim=True
            )

            # Optionally pad the anomaly maps to match input size
            if self.pad_maps:
                map_st = F.pad(map_st, (4, 4, 4, 4))
                map_stae = F.pad(map_stae, (4, 4, 4, 4))
            # Resize anomaly maps to original input size
            map_st = F.interpolate(map_st, size=(self.input_size[0], self.input_size[1]), mode="bilinear")
            map_stae = F.interpolate(map_stae, size=(self.input_size[0], self.input_size[1]), mode="bilinear")

            # Normalize anomaly maps if quantiles are set
            if self.is_set(self.quantiles) and normalize:
                map_st = 0.1 * (map_st - self.quantiles["qa_st"]) / (self.quantiles["qb_st"] - self.quantiles["qa_st"])
                map_stae = (
                    0.1 * (map_stae - self.quantiles["qa_ae"]) / (self.quantiles["qb_ae"] - self.quantiles["qa_ae"])
                )

            # Combine the anomaly maps from student and autoencoder
            map_combined = 0.5 * map_st + 0.5 * map_stae
            # Return the combined anomaly map
            return {"anomaly_map": map_combined, "map_st": map_st, "map_ae": map_stae}
