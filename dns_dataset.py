import os
import random
import torch
import torch.utils.data
import librosa
import glob
import numpy as np
import re

def extract_fileid_from_noisy_filename(filename):
    """Extract fileid from noisy filename like 'book_11346_chp_0030_reader_00812_56_XHdrRVey_w4-YR2AHUO0ISQ-8ubTow4S_6w_snr10_fileid_118096.wav'"""
    # Extract the fileid part from the filename
    match = re.search(r'fileid_(\d+)\.wav$', filename)
    if match:
        return f"clean_fileid_{match.group(1)}.wav"
    return None

def get_dns_dataset_filelist(training_file, validation_file):
    """Load DNS dataset file list with format: fileid|noisy_file_path"""
    with open(training_file, 'r', encoding='utf-8') as fi:
        training_pairs = [x.strip().split('|') for x in fi.read().split('\n') if len(x.strip()) > 0]

    with open(validation_file, 'r', encoding='utf-8') as fi:
        validation_pairs = [x.strip().split('|') for x in fi.read().split('\n') if len(x.strip()) > 0]

    return training_pairs, validation_pairs

class DNSDataset(torch.utils.data.Dataset):
    def __init__(self, training_pairs, clean_wavs_dir, segment_size, 
                sampling_rate, split=True, shuffle=True, n_cache_reuse=1, device=None):
        self.audio_pairs = training_pairs  # List of (fileid, noisy_file_path) tuples
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_pairs)
        self.clean_wavs_dir = clean_wavs_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.cached_clean_wav = None
        self.cached_noisy_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device

    def __getitem__(self, index):
        fileid, noisy_file_path = self.audio_pairs[index]
        
        if self._cache_ref_count == 0:
            # Load clean audio using fileid
            clean_audio, _ = librosa.load(os.path.join(self.clean_wavs_dir, fileid), sr=self.sampling_rate)
            # Load noisy audio using the full path
            noisy_audio, _ = librosa.load(noisy_file_path, sr=self.sampling_rate)
            
            length = min(len(clean_audio), len(noisy_audio))
            clean_audio, noisy_audio = clean_audio[: length], noisy_audio[: length]
            self.cached_clean_wav = clean_audio
            self.cached_noisy_wav = noisy_audio
            self._cache_ref_count = self.n_cache_reuse
        else:
            clean_audio = self.cached_clean_wav
            noisy_audio = self.cached_noisy_wav
            self._cache_ref_count -= 1
        
        clean_audio, noisy_audio = torch.FloatTensor(clean_audio), torch.FloatTensor(noisy_audio)
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor).unsqueeze(0)
        noisy_audio = (noisy_audio * norm_factor).unsqueeze(0)

        assert clean_audio.size(1) == noisy_audio.size(1)

        if self.split:
            if clean_audio.size(1) >= self.segment_size:
                max_audio_start = clean_audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                clean_audio = clean_audio[:, audio_start: audio_start+self.segment_size]
                noisy_audio = noisy_audio[:, audio_start: audio_start+self.segment_size]
            else:
                clean_audio = torch.nn.functional.pad(clean_audio, (0, self.segment_size - clean_audio.size(1)), 'constant')
                noisy_audio = torch.nn.functional.pad(noisy_audio, (0, self.segment_size - noisy_audio.size(1)), 'constant')

        return (clean_audio.squeeze(), noisy_audio.squeeze())

    def __len__(self):
        return len(self.audio_pairs) 