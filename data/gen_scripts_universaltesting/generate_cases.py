import os
import random
import configparser as CP
from pathlib import Path
import numpy as np
import librosa
from audiolib import audioread, audiowrite, snr_mixer
from scipy import signal
from sample_reverb import draw_params
from wham_room import WhamRoom
import pandas as pd

# -----------------------------------------------------------------------
# Config and parameters
# -----------------------------------------------------------------------
CFG_PATH = os.path.join(os.path.dirname(__file__), 'noisyspeech_synthesizer_multifunctional.cfg')
cfg = CP.ConfigParser()
cfg._interpolation = CP.ExtendedInterpolation()
cfg.read(CFG_PATH)
conf = cfg._sections['noisy_speech']

FS = int(conf.get('sampling_rate', 16000))
CLEAN_DIR = conf.get('speech_dir')
NOISE_DIR = conf.get('noise_dir')
OUT_ROOT = conf.get('noisy_destination') or os.path.join(os.path.dirname(__file__), 'out_noisy')
OUT_CLEAN_ROOT = conf.get('clean_destination') or os.path.join(os.path.dirname(__file__), 'out_clean')

random.seed(1)
np.random.seed(1)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def bandlimit_audio(audio, fs, cutoff_hz):
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return audio
    target_sr = int(min(fs, max(2 * int(cutoff_hz), 800)))

    if not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    
    # Downsample then Upsample
    y = librosa.resample(audio, orig_sr=fs, target_sr=target_sr, res_type='kaiser_best')
    y_up = librosa.resample(y, orig_sr=target_sr, target_sr=fs, res_type='kaiser_best')
    return y_up

def get_amplitude_scalar(audio, target_level_db):
    """Calculates the scalar factor needed to reach target_level_db RMS."""
    if not np.issubdtype(audio.dtype, np.floating):
        audio = audio.astype(np.float32)
    rms = np.sqrt(np.mean(audio ** 2))
    scalar = 10 ** (target_level_db / 20) / (rms + 1e-13)
    return scalar

# -----------------------------------------------------------------------
# Collect files
# -----------------------------------------------------------------------
clean_files = [str(p) for p in Path(CLEAN_DIR).rglob('*.wav')]
noise_files = [str(p) for p in Path(NOISE_DIR).rglob('*.wav')]
if len(clean_files) == 0:
    raise RuntimeError('No clean files found under configured paths')

# params for snr_mixer and manual scaling
params = {'cfg': conf,
          'target_level_lower': int(conf.get('target_level_lower', -35)),
          'target_level_upper': int(conf.get('target_level_upper', -15))}

reverb_param_df = pd.read_csv(os.path.join('data', 'reverb_params_tt.csv'))

os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(OUT_CLEAN_ROOT, exist_ok=True)

# -----------------------------------------------------------------------
# Task Definitions
# -----------------------------------------------------------------------
tasks = [
    # 1. Only Reverb (250 total)
    {
        'name': 'only_reverb',
        'snrs': [None],
        'count_per_snr': 250,
        'reverb': True, 'noise': False, 'limit_freq': None
    },
    # 2. Only Bandlimit (250 total -> split into 125 @ 2k, 125 @ 4k)
    {
        'name': 'only_bandlimit',
        'subcases': [
            {'sub_name': '2khz', 'cutoff': 2000, 'count': 125},
            {'sub_name': '4khz', 'cutoff': 4000, 'count': 125}
        ],
        'reverb': False, 'noise': False
    },
    # 3. Only Noise (5 SNRs * 50 = 250 total)
    {
        'name': 'only_noise',
        'snrs': [-5, 0, 5, 10, 15],
        'count_per_snr': 50,
        'reverb': False, 'noise': True, 'limit_freq': None
    },
    # 4. Noise + Reverb (5 SNRs * 50 = 250 total)
    {
        'name': 'noise_reverb',
        'snrs': [-5, 0, 5, 10, 15],
        'count_per_snr': 50,
        'reverb': True, 'noise': True, 'limit_freq': None
    },
    # 5. Noise + Bandlimit (5 SNRs * 50 = 250 total, always 4kHz)
    {
        'name': 'noise_limit',
        'snrs': [-5, 0, 5, 10, 15],
        'count_per_snr': 50,
        'reverb': False, 'noise': True, 'limit_freq': 4000
    },
    # 6. Noise + Reverb + Bandlimit (5 SNRs * 50 = 250 total, always 4kHz)
    {
        'name': 'noise_reverb_limit',
        'snrs': [-5, 0, 5, 10, 15],
        'count_per_snr': 50,
        'reverb': True, 'noise': True, 'limit_freq': 4000
    }
]

# -----------------------------------------------------------------------
# Processing Loop
# -----------------------------------------------------------------------

for task in tasks:
    if 'subcases' in task:
        sub_tasks = []
        for sc in task['subcases']:
            st = task.copy()
            st['name'] = os.path.join(task['name'], sc['sub_name'])
            st['snrs'] = [None]
            st['count_per_snr'] = sc['count']
            st['limit_freq'] = sc['cutoff']
            del st['subcases']
            sub_tasks.append(st)
        loop_tasks = sub_tasks
    else:
        loop_tasks = [task]

    for current_task in loop_tasks:
        case_name = current_task['name']
        snrs = current_task['snrs']
        count_target = current_task['count_per_snr']
        
        has_reverb = current_task['reverb']
        has_noise = current_task['noise']
        limit_freq = current_task['limit_freq']

        for snr in snrs:
            if snr is not None:
                out_dir = os.path.join(OUT_ROOT, case_name, f"{snr}db")
                out_clean_dir = os.path.join(OUT_CLEAN_ROOT, case_name, f"{snr}db")
            else:
                out_dir = os.path.join(OUT_ROOT, case_name)
                out_clean_dir = os.path.join(OUT_CLEAN_ROOT, case_name)

            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(out_clean_dir, exist_ok=True)

            generated = 0
            tries = 0
            
            print(f"Starting Task: {case_name}, SNR: {snr}, Target: {count_target}")

            while generated < count_target and tries < count_target * 20:
                tries += 1
                
                # 1. Load Clean
                clean_path = random.choice(clean_files)
                clean, fs_c = audioread(clean_path)
                if fs_c != FS:
                    clean = librosa.resample(clean, orig_sr=fs_c, target_sr=FS)

                clean_proc = clean.copy()
                clean_ref = clean.copy() 

                # 2. Apply Reverb (FIRST)
                if has_reverb:
                    idx = np.random.randint(0, len(reverb_param_df))
                    utt_row = reverb_param_df.iloc[idx] 
                    room = WhamRoom(p=[utt_row['room_x'], utt_row['room_y'], utt_row['room_z']],
                                    mics=[[utt_row['micL_x'], utt_row['micL_y'], utt_row['mic_z']]],
                                    s1=[utt_row['s1_x'], utt_row['s1_y'], utt_row['s1_z']],
                                    T60=utt_row['T60'])
                    room.generate_rirs()
                    room.add_audio(clean_proc)
                    
                    clean_proc_rev = room.generate_audio(anechoic=False, fs=FS)
                    clean_ref_rev = room.generate_audio(anechoic=True, fs=FS)

                    clean_proc = clean_proc_rev[0][0]
                    clean_ref = clean_ref_rev[0][0]

                    length = min(len(clean_proc), len(clean_ref))
                    clean_proc = clean_proc[:length]
                    clean_ref = clean_ref[:length]

                # 3. Apply Noise (SECOND)
                if has_noise:
                    if len(noise_files) == 0:
                        print("Warning: No noise files but noise requested.")
                        continue
                        
                    noise_path = random.choice(noise_files)
                    noise, fs_n = audioread(noise_path)
                    if fs_n != FS:
                        noise = librosa.resample(noise, orig_sr=fs_n, target_sr=FS)

                    target_len = len(clean_proc)
                    if len(noise) < target_len:
                        reps = int(np.ceil(target_len / len(noise)))
                        noise = np.tile(noise, reps)[:target_len]
                    else:
                        start = random.randint(0, len(noise) - target_len)
                        noise = noise[start:start + target_len]
                    tgt_lvl = random.randint(params['target_level_lower'], params['target_level_upper'])

                    # snr_mixer handles normalization and mixing
                    _, _, noisy_out, tl, clean_out = snr_mixer(
                        params=params, 
                        clean=clean_proc, 
                        noise=noise, 
                        refclean=clean_ref, 
                        target_level=tgt_lvl,
                        snr=snr
                    )
                else:
                    # No Noise: Manual Normalization
                    tgt_lvl = random.randint(params['target_level_lower'], params['target_level_upper'])
                    
                    # CORRECTION: Calculate scalar from input (clean_proc) and apply SAME scalar to both
                    scalar = get_amplitude_scalar(clean_proc, tgt_lvl)
                    
                    noisy_out = clean_proc * scalar
                    clean_out = clean_ref * scalar

                # 4. Apply Bandlimit (LAST)
                if limit_freq is not None:
                    # Apply ONLY to the noisy/input signal
                    noisy_out = bandlimit_audio(noisy_out, FS, limit_freq)

                # Check clipping
                max_amp = max(np.max(np.abs(noisy_out)), np.max(np.abs(clean_out)))
                if max_amp > 1.0:
                    noisy_out = noisy_out / (max_amp + 1e-6)
                    clean_out = clean_out / (max_amp + 1e-6)

                # Save
                fname = f"file_{generated:03d}"
                if snr is not None:
                    fname += f"_snr{snr}"
                fname += ".wav"

                out_path = os.path.join(out_dir, fname)
                out_clean_path = os.path.join(out_clean_dir, fname)
                
                audiowrite(out_path, noisy_out, FS)
                audiowrite(out_clean_path, clean_out, FS)
                
                generated += 1
                
                if generated % 50 == 0:
                    print(f"  Generated {generated}/{count_target}")

print('All tasks completed.')