import numpy as np
import os
import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn

import config as c
from freia_funcs import *

MODEL_DIR = './models'

# Function to create a normalizing flow model for the teacher.
def get_nf(input_dim=c.n_feat, channels_hidden=c.channels_hidden_teacher):
    nodes = list()
    # If positional encoding is enabled, add an input node for it.
    if c.pos_enc:
        nodes.append(InputNode(c.pos_enc_dim, name='input'))
    # Main input node.
    nodes.append(InputNode(input_dim, name='input'))
    # Creating coupling blocks.
    for k in range(c.n_coupling_blocks):
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # Conditional coupling layer if positional encoding is used.
        if c.pos_enc:
            nodes.append(Node([nodes[-1].out0, nodes[0].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv,
                               'cond_dim': c.pos_enc_dim,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
        # Standard coupling layer.
        else:
            nodes.append(Node([nodes[-1].out0], glow_coupling_layer_cond,
                              {'clamp': c.clamp,
                               'F_class': F_conv,
                               'F_args': {'channels_hidden': channels_hidden,
                                          'kernel_size': c.kernel_sizes[k]}},
                              name=F'conv_{k}'))
    # Output node.
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    # Creating the reversible graph net.
    nf = ReversibleGraphNet(nodes, n_jac=1)
    return nf

# A module to extract features from a specified layer of EfficientNet.
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
    return P.to(c.device)[None]

# The main model class combining feature extraction, normalizing flow, and positional encoding.
class Model(nn.Module):
    def __init__(self, nf=True, n_blocks=c.n_coupling_blocks, channels_hidden=c.channels_hidden_teacher):
        super(Model, self).__init__()

        # Conditional initialization for the feature extractor.
        if not c.pre_extracted:
            self.feature_extractor = FeatureExtractor(layer_idx=c.extract_layer)

        if nf:
            self.net = get_nf()
        else:
            self.net = Student(channels_hidden=channels_hidden, n_blocks=n_blocks)

        # Positional encoding initialization if enabled.
        if c.pos_enc:
            self.pos_enc = positionalencoding2d(c.pos_enc_dim, c.map_len, c.map_len)

        # Unshuffle operation for processing depth information.
        self.unshuffle = nn.PixelUnshuffle(c.depth_downscale)

    def forward(self, x, depth):
        # Feature extraction based on the mode and configuration.
        if not c.pre_extracted and c.mode != 'depth':
            with torch.no_grad():
                f = self.feature_extractor(x)
        else:
            f = x

        # Conditional input preparation based on the mode.
        if c.mode == 'RGB':
            inp = f
        elif c.mode == 'depth':
            inp = self.unshuffle(depth)
        elif c.mode == 'combi':
            inp = torch.cat([f, self.unshuffle(depth)], dim=1)
        else:
            raise RuntimeError('no valid mode selected, choose from {\'RGB\', \'depth\', \'combi\'}')

        # Processing through the network with or without positional encoding.
        if c.pos_enc:
            cond = self.pos_enc.tile(inp.shape[0], 1, 1, 1)
            z = self.net([cond, inp])
        else:
            z = self.net(inp)
        # Calculating the Jacobian for the normalizing flow.
        jac = self.net.jacobian(run_forward=False)[0]
        # Returning the transformed input and Jacobian.
        return z, jac

# A residual block used in the student model for processing features.
class res_block(nn.Module):
    def __init__(self, channels):
        super(res_block, self).__init__()
        self.l1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.l2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        inp = x
        x = self.l1(x)
        x = self.bn1(x)
        x = self.act(x)

        x = self.l2(x)
        x = self.bn2(x)
        x = self.act(x)
        # Applying residual connection.
        x = x + inp
        return x

# The student model consisting of convolutional layers and residual blocks.
class Student(nn.Module):
    def __init__(self, channels_hidden=c.channels_hidden_student, n_blocks=c.n_st_blocks):
        super(Student, self).__init__()
        inp_feat = c.n_feat if not c.pos_enc else c.n_feat + c.pos_enc_dim
        self.conv1 = nn.Conv2d(inp_feat, channels_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels_hidden, c.n_feat, kernel_size=3, padding=1)
        self.res = list()
        # Initializing residual blocks.
        for _ in range(n_blocks):
            self.res.append(res_block(channels_hidden))
        self.res = nn.ModuleList(self.res)
        # Learnable parameter for scaling.
        self.gamma = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU()

    def forward(self, x):
        # Concatenating positional encoding if enabled.
        if c.pos_enc:
            x = torch.cat(x, dim=1)

        x = self.act(self.conv1(x))
        # Passing through residual blocks.
        for i in range(len(self.res)):
            x = self.res[i](x)

        x = self.conv2(x)
        return x

    def jacobian(self, run_forward=False):
        return [0] # Placeholder for Jacobian computation, not applicable for student model.

# Functions for saving and loading model weights.
def save_weights(model, suffix):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    model.to('cpu')
    torch.save(model.net.state_dict(), join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth'))
    print('student saved')
    model.to(c.device)


def load_weights(model, suffix):
    model.net.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'{c.modelname}_{c.class_name}_{suffix}.pth')))
    model.eval()
    model.to(c.device)
    return model
