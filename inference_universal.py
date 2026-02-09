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

    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    # --- MODIFIED: Recursive Scan ---
    # Use glob to find all .wav files recursively (handles noisy/-5db/file.wav)
    print(f"Scanning for wav files in {a.input_noisy_wavs_dir}...")
    all_files = sorted(glob.glob(os.path.join(a.input_noisy_wavs_dir, '**', '*.wav'), recursive=True))
    
    if not all_files:
        print("No wav files found! Check the input directory path.")
        return

    print(f"Found {len(all_files)} files to process.")

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for file_path in track(all_files):
            # Load using the full path found by glob
            noisy_wav, _ = librosa.load(file_path, sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            # --- MODIFIED: Output Path Calculation ---
            # 1. Calculate relative path (e.g., "noisy/-5db/file_002.wav")
            rel_path = os.path.relpath(file_path, a.input_noisy_wavs_dir)
            
            # 2. Join with output root (e.g., "generated/noisy/-5db/file_002.wav")
            output_file = os.path.join(a.output_dir, rel_path)

            # 3. Create the specific subdirectory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/data2/wangchengzhong/challenge/noisy_test')
    parser.add_argument('--output_dir', default='/data2/wangchengzhong/challenge/enh')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    # Load config relative to checkpoint directory
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config_universal.json')

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