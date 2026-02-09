import os
import glob
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pylab as plt


import torch.nn.functional as F
import numpy as np

def anti_wrapping_function(y_true, y_pred):
    diff = y_pred - y_true
    return diff - 2 * np.pi * torch.round(diff / (2 * np.pi))

class WeightedOmnidirectionalPhaseLoss(nn.Module):
    def __init__(self):
        super(WeightedOmnidirectionalPhaseLoss, self).__init__()
        # Define kernels
        kernels_np = np.array([
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]], 
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]], 
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]], 
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]], 
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=np.float32)
        
        kernels = torch.from_numpy(kernels_np).unsqueeze(1) # Add in_channels dim
        
        # Register as a buffer so it moves with the model (e.g., .to(device))
        self.register_buffer('kernels', kernels)

    def forward(self, true_phase, pred_phase, true_magnitude):
        # Reshape for conv2d
        true_phase = true_phase.unsqueeze(1)
        pred_phase = pred_phase.unsqueeze(1)

        # self.kernels will already be on the correct device
        delta_p_true = F.conv2d(true_phase, self.kernels, padding='same')
        delta_p_pred = F.conv2d(pred_phase, self.kernels, padding='same')

        loss_map = torch.abs(anti_wrapping_function(delta_p_true, delta_p_pred))
        
        weighted_loss_map = true_magnitude.unsqueeze(1) * loss_map

        return torch.mean(weighted_loss_map)





        
def plot_spectrogram(spectrogram):
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    fig.canvas.draw()
    plt.close()

    return fig


def get_padding(kernel_size, dilation=1):
    return int((kernel_size*dilation - dilation)/2)


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0]*dilation[0] - dilation[0])/2), int((kernel_size[1]*dilation[1] - dilation[1])/2))


class LearnableSigmoid1d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)


class Sigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = torch.ones(in_features, 1)

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

class PLSigmoid(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(in_features, 1) * 2.0)
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.beta.requiresGrad = True
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
    

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def save_checkpoint(filepath, obj):
    print("Saving checkpoint to {}".format(filepath))
    torch.save(obj, filepath)
    print("Complete.")


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '????????')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return None
    return sorted(cp_list)[-1]

def remove_older_checkpoint(filepath, pre='g', max_to_keep=5):
    par_file_dir, filename = os.path.split(filepath)
    if os.path.exists(os.path.join(par_file_dir, f'checkpoint_{pre}')):
        with open(os.path.join(par_file_dir, f'checkpoint_{pre}'), 'r') as f:
            ckpts = []
            for y in f.readlines():
                ckpts.append(y.split()[0])
    else:
        ckpts = []
    ckpts.append(filename)
    for item in ckpts[: -max_to_keep]:
        if os.path.exists(os.path.join(par_file_dir, item)):
            os.remove(os.path.join(par_file_dir, item))
    with open(os.path.join(par_file_dir, f'checkpoint_{pre}'), 'w') as f:
        for item in ckpts[-max_to_keep:]:
            f.write("{}\n".format(item))

def main():
    plsigmoid = PLSigmoid(201)
    a = torch.randn(4, 201, 100)
    print(plsigmoid(a))

if __name__ == '__main__':
    main()
