from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = MPNet(h).to(device)
    onnx_model_path = "dual_input_model.onnx"


    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    test_indexes = os.listdir(a.input_noisy_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    # print("🚀 Starting ONNX export for dual-input model...")
    # dummy_mag_input = torch.randn(1, 201, 100).to(device)
    # dummy_pha_input = torch.randn(1, 201, 100).to(device)
    # torch.onnx.export(
    #     model,
    #     (dummy_mag_input, dummy_pha_input),  # Pass dummy inputs as a TUPLE
    #     onnx_model_path,
    #     export_params=True,
    #     opset_version=13,
    #     do_constant_folding=True,
    #     input_names=['noisy_mag', 'noisy_pha'],  # List of input names in order
    #     output_names=['output'],
    #     dynamic_axes={
    #         'noisy_mag': {2: 'time_steps'},  # Dynamic B and T for the first input
    #         'noisy_pha': {2: 'time_steps'},  # Dynamic B and T for the second input
    #         'output': {2: 'time_steps'}    # Dynamic B and T for the output
    #     }
    # )
    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, torch.zeros_like(noisy_pha).to(noisy_pha.device))
            audio_g = mag_pha_istft(noisy_amp, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index.split(".wav")[0] + a.name + '.wav')

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/data/home/wangchengzhong/dtb/VBD/VBD_real/test/clean') # change this to your VBD clean dir
    parser.add_argument('--output_dir', default='./generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--name', default='',required=False)

    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config_small.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
