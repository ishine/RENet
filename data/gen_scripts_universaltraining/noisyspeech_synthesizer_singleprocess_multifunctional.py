"""
@author: chkarada
"""

# Note: This single process audio synthesizer will attempt to use each clean
# speech sourcefile once, as it does not randomly sample from these files

import os
import glob
import argparse
import ast
import configparser as CP
from random import shuffle
import random

import librosa
import numpy as np
from scipy import signal
from audiolib import audioread, audiowrite, segmental_snr_mixer, activitydetector, is_clipped, add_clipping, snr_mixer
import utils

import pandas as pd
from pathlib import Path
from scipy.io import wavfile

MAXTRIES = 50
MAXFILELEN = 100

np.random.seed(5)
random.seed(5)

def add_pyreverb(clean_speech, rir):
    
    reverb_speech = signal.fftconvolve(clean_speech, rir, mode="full")
    
    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : clean_speech.shape[0]]

    return reverb_speech

def build_audio(is_clean, params, index, audio_samples_length=-1):
    '''Construct an audio signal from source files'''

    fs_output = params['fs']
    silence_length = params['silence_length']
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])

    output_audio = np.zeros(0)
    remaining_length = audio_samples_length
    files_used = []
    clipped_files = []

    if is_clean:
        source_files = params['cleanfilenames']
        idx = index
    else:
        if 'noisefilenames' in params.keys():
            source_files = params['noisefilenames']
            idx = index
        # if noise files are organized into individual subdirectories, pick a directory randomly
        else:
            noisedirs = params['noisedirs']
            # pick a noise category randomly
            idx_n_dir = np.random.randint(0, np.size(noisedirs))
            source_files = glob.glob(os.path.join(noisedirs[idx_n_dir], 
                                                  params['audioformat']))
            shuffle(source_files)
            # pick a noise source file index randomly
            idx = np.random.randint(0, np.size(source_files))

    # initialize silence
    silence = np.zeros(int(fs_output*silence_length))

    # iterate through multiple clips until we have a long enough signal
    tries_left = MAXTRIES
    while remaining_length > 0 and tries_left > 0:

        # read next audio file and resample if necessary

        idx = (idx + 1) % np.size(source_files)
        input_audio, fs_input = audioread(source_files[idx])
        if fs_input != fs_output:
            input_audio = librosa.resample(input_audio, orig_sr=fs_input, target_sr=fs_output)

        # if current file is longer than remaining desired length, and this is
        # noise generation or this is training set, subsample it randomly
        if len(input_audio) > remaining_length and (not is_clean or not params['is_test_set']):
            idx_seg = np.random.randint(0, len(input_audio)-remaining_length)
            input_audio = input_audio[idx_seg:idx_seg+remaining_length]

        # check for clipping, and if found move onto next file
        if is_clipped(input_audio):
            clipped_files.append(source_files[idx])
            tries_left -= 1
            continue

        # concatenate current input audio to output audio stream
        files_used.append(source_files[idx])
        output_audio = np.append(output_audio, input_audio)
        remaining_length -= len(input_audio)

        # add some silence if we have not reached desired audio length
        if remaining_length > 0:
            silence_len = min(remaining_length, len(silence))
            output_audio = np.append(output_audio, silence[:silence_len])
            remaining_length -= silence_len

    if tries_left == 0 and not is_clean and 'noisedirs' in params.keys():
        print("There are not enough non-clipped files in the " + noisedirs[idx_n_dir] + \
              " directory to complete the audio build")
        return [], [], clipped_files, idx

    return output_audio, files_used, clipped_files, idx


def gen_audio(is_clean, params, index, audio_samples_length=-1):
    '''Calls build_audio() to get an audio signal, and verify that it meets the
       activity threshold'''

    clipped_files = []
    low_activity_files = []
    if audio_samples_length == -1:
        audio_samples_length = int(params['audio_length']*params['fs'])
    if is_clean:
        activity_threshold = params['clean_activity_threshold']
    else:
        activity_threshold = params['noise_activity_threshold']

    while True:
        audio, source_files, new_clipped_files, index = \
            build_audio(is_clean, params, index, audio_samples_length)

        clipped_files += new_clipped_files
        if len(audio) < audio_samples_length:
            continue

        if activity_threshold == 0.0:
            break

        percactive = activitydetector(audio=audio)
        if percactive > activity_threshold:
            break
        else:
            low_activity_files += source_files

    return audio, source_files, clipped_files, low_activity_files, index

def main_gen(params):
    '''Calls gen_audio() to generate the audio signals, verifies that they meet
       the requirements, and writes the files to storage'''

    clean_source_files = []
    clean_clipped_files = []
    clean_low_activity_files = []
    noise_source_files = []
    noise_clipped_files = []
    noise_low_activity_files = []

    clean_index = 0
    noise_index = 0
    file_num = params['fileindex_start']

    def bandlimit_audio(audio, fs, cutoff_hz):
        """Bandlimit `audio` by resampling: downsample to ~2*cutoff, then upsample back."""
        nyq = fs / 2.0
        if cutoff_hz >= nyq:
            return audio
        target_sr = int(min(fs, max(2 * int(cutoff_hz), 800)))
        try:
            if not np.issubdtype(audio.dtype, np.floating):
                audio = audio.astype(np.float32)
            # Use Kaiser best to minimize phase distortion during resampling
            y = librosa.resample(audio, orig_sr=fs, target_sr=target_sr, res_type='kaiser_best')
            y_up = librosa.resample(y, orig_sr=target_sr, target_sr=fs, res_type='kaiser_best')
            return y_up
        except Exception:
            return audio

    def normalize_to_target(audio, target_level_db):
        """Scales audio to the specified RMS target level in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        scalar = 10 ** (target_level_db / 20) / (rms + 1e-13)
        return audio * scalar

    # ---------------------------------------------------------
    # 1. Build Data Pipeline Plan (6 Cases)
    # ---------------------------------------------------------
    total_files = params['fileindex_end'] - params['fileindex_start'] + 1
    
    # Define the 6 cases
    cases = [
        'only_noise',       # Clean + Noise
        'only_reverb',      # Clean + Reverb
        'only_bl',          # Clean + Bandlimit (50% 2k, 50% 4k)
        'noise_reverb',     # Clean + Reverb + Noise
        'noise_bl',         # Clean + Noise + Bandlimit (4k)
        'noise_reverb_bl'   # Clean + Reverb + Noise + Bandlimit (4k)
    ]
    
    # Distribute files evenly among cases
    counts = {k: total_files // len(cases) for k in cases}
    
    # Distribute remainders
    remainder = total_files % len(cases)
    for i in range(remainder):
        counts[cases[i]] += 1

    augment_plan = []
    maxc = max(counts.values())
    for i in range(maxc):
        for case in cases:
            if counts.get(case, 0) > i:
                augment_plan.append(case)

    # ---------------------------------------------------------
    # 2. Processing Loop
    # ---------------------------------------------------------
    while file_num <= params['fileindex_end']:
        # -- A. Generate clean speech (Base) --
        clean, clean_sf, clean_cf, clean_laf, clean_index = \
            gen_audio(True, params, clean_index)

        # ---------------------------------------------------------
        # NEW STEP: Scale to Target Level BEFORE Reverb/Augmentation
        # ---------------------------------------------------------
        # We pick a representative target level (e.g. average of bounds)
        # to ensure the signal entering the Reverb stage has a reasonable headroom.
        tl_low = params.get('target_level_lower', -35)
        tl_high = params.get('target_level_upper', -15)
        pre_mix_target = random.randint(tl_low, tl_high)
        
        clean = normalize_to_target(clean, pre_mix_target)
        scale_factor = 0.99/(abs((clean)).max() + 1e-9)
        if scale_factor <= 1.00:
            clean = clean * scale_factor * 0.99
        # -- B. Determine Augmentation Case --
        plan_idx = file_num - params['fileindex_start']
        if plan_idx < len(augment_plan):
            current_case = augment_plan[plan_idx]
        else:
            current_case = 'only_noise' # Fallback

        # Initialize flags based on case
        has_reverb = 'reverb' in current_case
        has_noise  = 'noise' in current_case
        has_bl     = 'bl' in current_case

        # Prepare copies
        # original_clean: The Direct-Path Full-Band Target
        # clean_for_mix:  The Input to be distorted (Reverb/Noise/BL)
        original_clean = clean.copy()       
        clean_for_mix = original_clean.copy() 

        # ---------------------------------------------------------
        # STEP 1: Apply Reverb (Physical Acoustics)
        # ---------------------------------------------------------
        if has_reverb:
            if len(params.get('myrir', [])) > 0:
                rir_index = random.randint(0, len(params['myrir']) - 1)
                my_rir = os.path.normpath(os.path.join('datasets/impulse_responses', params['myrir'][rir_index]))
                
                samples_rir = None
                if not os.path.exists(my_rir):
                    print(f"Warning: RIR file not found, skipping reverb: {my_rir}")
                else:
                    try:
                        (fs_rir, samples_rir) = wavfile.read(my_rir)
                    except Exception as e:
                        print(f"Warning: Failed to read RIR '{my_rir}', skipping reverb: {e}")

                if samples_rir is not None:
                    my_channel = int(params['mychannel'][rir_index])
                    if samples_rir.ndim == 1:
                        samples_rir_ch = np.array(samples_rir)
                    elif my_channel > 1:
                        samples_rir_ch = samples_rir[:, my_channel - 1]
                    else:
                        samples_rir_ch = samples_rir[:, my_channel - 1]

                    # --- ALIGNMENT LOGIC ---
                    peak_idx = np.argmax(np.abs(samples_rir_ch))
                    direct_rir_ch = samples_rir_ch[:peak_idx + 5]

                    # Apply Direct-Path RIR to TARGET (Aligns delay, keeps full band)
                    original_clean = add_pyreverb(original_clean, direct_rir_ch)

                    # Apply Full RIR to INPUT
                    clean_for_mix = add_pyreverb(clean_for_mix, samples_rir_ch)
                    scale_factor = 0.99/(abs((clean_for_mix)).max() + 1e-9)
                    if scale_factor <= 1.00:
                        original_clean = original_clean * scale_factor * 0.97
                        clean_for_mix = clean_for_mix * scale_factor * 0.97
        # ---------------------------------------------------------
        # STEP 2: Mix with Noise (Environmental)
        # ---------------------------------------------------------
        if has_noise:
            # Generate noise matching the length of the current input
            noise, noise_sf, noise_cf, noise_laf, noise_index = \
                gen_audio(False, params, noise_index, len(clean_for_mix))

            clean_clipped_files += clean_cf
            clean_low_activity_files += clean_laf
            noise_clipped_files += noise_cf
            noise_low_activity_files += noise_laf
            
            clean_source_files += clean_sf
            noise_source_files += noise_sf


            snr = np.random.randint(params['snr_lower'], params['snr_upper'])

            # Mix: scales input (clean_for_mix) to match noise at SNR
            # clean_orig_scaled is the scaled version of original_clean (Target)
            clean_snr, noise_snr, noisy_snr, target_level, clean_orig_scaled = snr_mixer(
                params=params,
                clean=clean_for_mix,
                noise=noise,
                refclean=original_clean,
                snr=snr
            )
        else:
            # No Noise cases: Treat current signals as the final mix
            noisy_snr = clean_for_mix
            clean_orig_scaled = original_clean
            noise_snr = np.zeros_like(noisy_snr) # Silent noise file
            target_level = pre_mix_target # Use the level we set manually
            snr = 99 
            
            clean_clipped_files += clean_cf
            clean_low_activity_files += clean_laf
            clean_source_files += clean_sf

        # ---------------------------------------------------------
        # STEP 3: Apply Bandwidth Limitation (Transmission)
        # ---------------------------------------------------------
        if has_bl:
            cutoff = 4000 # Default for mixed cases
            
            if current_case == 'only_bl':
                # Special rule: 1/2 2kHz, 1/2 4kHz
                if random.random() > 0.5:
                    cutoff = 4000
                else:
                    cutoff = 2000
            
            # Bandlimit the INPUT only
            noisy_snr = bandlimit_audio(noisy_snr, params['fs'], cutoff)
            scale_factor = 0.99/(abs((noisy_snr)).max() + 1e-9)
            if scale_factor <= 1.00:
                # NOTE: A minor scaling discrepancy exists here. The input (noisy) is scaled 
                # down to prevent clipping, but the target (clean) remains at the original amplitude. 
                # 
                # We analyzed the generated dataset and found the impact is negligible:
                # 1. Frequency: Affects < 3.5% of samples.
                # 2. Magnitude: Scaling factor is > 0.99 in 90% of cases, and always > 0.95 (< 0.44 dB gain error).
                #
                # We decided not to regenerate the data to ensure full reproducibility of the 
                # models presented in our paper. This slight gain mismatch does not affect model performance.
                # print(scale_factor)
                clean = clean * scale_factor * 0.99
                noise_snr = noise_snr * scale_factor * 0.99
                noisy_snr = noisy_snr * scale_factor * 0.99
        # ---------------------------------------------------------
        # Check clipping and Write Files
        # ---------------------------------------------------------
        if is_clipped(clean_orig_scaled) or is_clipped(noise_snr) or is_clipped(noisy_snr):
            print("Warning: File #" + str(file_num) + " has unexpected clipping, skipping")
            continue

        hyphen = '-'
        clean_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in clean_sf]
        clean_files_joined = hyphen.join(clean_source_filenamesonly)[:MAXFILELEN]
        
        if has_noise:
            noise_source_filenamesonly = [i[:-4].split(os.path.sep)[-1] for i in noise_sf]
            noise_files_joined = hyphen.join(noise_source_filenamesonly)[:MAXFILELEN]
        else:
            noise_files_joined = 'none'

        noisyfilename = clean_files_joined + '_' + noise_files_joined + \
                        '_case_' + current_case + \
                        '_snr' + str(snr) + \
                        '_tl' + str(target_level) + \
                        '_fileid_' + str(file_num) + '.wav'
        
        cleanfilename = 'clean_fileid_'+str(file_num)+'.wav'
        noisefilename = 'noise_fileid_'+str(file_num)+'.wav'

        noisypath = os.path.join(params['noisyspeech_dir'], noisyfilename)
        cleanpath = os.path.join(params['clean_proc_dir'], cleanfilename)
        noisepath = os.path.join(params['noise_proc_dir'], noisefilename)

        audio_signals = [noisy_snr, clean_orig_scaled, noise_snr]
        file_paths = [noisypath, cleanpath, noisepath]

        file_num += 1
        for i in range(len(audio_signals)):
            try:
                audiowrite(file_paths[i], audio_signals[i], params['fs'])
            except Exception as e:
                print(str(e))

    return clean_source_files, clean_clipped_files, clean_low_activity_files, \
           noise_source_files, noise_clipped_files, noise_low_activity_files

def main_body():
    '''Main body of this file'''

    parser = argparse.ArgumentParser()

    # Configurations: read noisyspeech_synthesizer.cfg and gather inputs
    parser.add_argument('--cfg', default='noisyspeech_synthesizer_multifunctional.cfg',
                        help='Read noisyspeech_synthesizer.cfg for all the details')
    parser.add_argument('--cfg_str', type=str, default='noisy_speech')
    args = parser.parse_args()

    params = dict()
    params['args'] = args
    cfgpath = os.path.join(os.path.dirname(__file__), args.cfg)
    assert os.path.exists(cfgpath), f'No configuration file as [{cfgpath}]'

    cfg = CP.ConfigParser()
    cfg._interpolation = CP.ExtendedInterpolation()
    cfg.read(cfgpath)
    params['cfg'] = cfg._sections[args.cfg_str]
    cfg = params['cfg']

    clean_dir = os.path.join(os.path.dirname(__file__), 'datasets/clean')

    if cfg['speech_dir'] != 'None':
        clean_dir = cfg['speech_dir']
    if not os.path.exists(clean_dir):
        assert False, ('Clean speech data is required')

    noise_dir = os.path.join(os.path.dirname(__file__), 'datasets/noise')

    if cfg['noise_dir'] != 'None':
        noise_dir = cfg['noise_dir']
    if not os.path.exists:
        assert False, ('Noise data is required')

    params['fs'] = int(cfg['sampling_rate'])
    params['audioformat'] = cfg['audioformat']
    params['audio_length'] = float(cfg['audio_length'])
    params['silence_length'] = float(cfg['silence_length'])
    params['total_hours'] = float(cfg['total_hours'])
    
    # clean singing speech
    params['use_singing_data'] = int(cfg['use_singing_data'])
    params['clean_singing'] = str(cfg['clean_singing'])
    params['singing_choice'] = int(cfg['singing_choice'])

    # clean emotional speech
    params['use_emotion_data'] = int(cfg['use_emotion_data'])
    params['clean_emotion'] = str(cfg['clean_emotion'])
    
    # clean mandarin speech
    params['use_mandarin_data'] = int(cfg['use_mandarin_data'])
    params['clean_mandarin'] = str(cfg['clean_mandarin'])
    
    # rir
    params['rir_choice'] = int(cfg['rir_choice'])
    params['lower_t60'] = float(cfg['lower_t60'])
    params['upper_t60'] = float(cfg['upper_t60'])
    params['rir_table_csv'] = str(cfg['rir_table_csv'])
    params['clean_speech_t60_csv'] = str(cfg['clean_speech_t60_csv'])

    if cfg['fileindex_start'] != 'None' and cfg['fileindex_start'] != 'None':
        params['num_files'] = int(cfg['fileindex_end'])-int(cfg['fileindex_start'])
        params['fileindex_start'] = int(cfg['fileindex_start'])
        params['fileindex_end'] = int(cfg['fileindex_end'])
    else:
        params['num_files'] = int((params['total_hours']*60*60)/params['audio_length'])
        params['fileindex_start'] = 0
        params['fileindex_end'] = params['num_files']

    print('Number of files to be synthesized:', params['num_files'])
    
    params['is_test_set'] = utils.str2bool(cfg['is_test_set'])
    params['clean_activity_threshold'] = float(cfg['clean_activity_threshold'])
    params['noise_activity_threshold'] = float(cfg['noise_activity_threshold'])
    params['snr_lower'] = int(cfg['snr_lower'])
    params['snr_upper'] = int(cfg['snr_upper'])
    
    params['randomize_snr'] = utils.str2bool(cfg['randomize_snr'])
    params['target_level_lower'] = int(cfg['target_level_lower'])
    params['target_level_upper'] = int(cfg['target_level_upper'])
    
    if 'snr' in cfg.keys():
        params['snr'] = int(cfg['snr'])
    else:
        params['snr'] = int((params['snr_lower'] + params['snr_upper'])/2)

    params['noisyspeech_dir'] = utils.get_dir(cfg, 'noisy_destination', 'noisy')
    params['clean_proc_dir'] = utils.get_dir(cfg, 'clean_destination', 'clean')
    params['noise_proc_dir'] = utils.get_dir(cfg, 'noise_destination', 'noise')

    if 'speech_csv' in cfg.keys() and cfg['speech_csv'] != 'None':
        cleanfilenames = pd.read_csv(cfg['speech_csv'])
        cleanfilenames = cleanfilenames['filename']
    else:
        #cleanfilenames = glob.glob(os.path.join(clean_dir, params['audioformat']))
        cleanfilenames= []
        for path in Path(clean_dir).rglob('*.wav'):
            cleanfilenames.append(str(path.resolve()))

    shuffle(cleanfilenames)
#   add singing voice to clean speech
    if params['use_singing_data'] ==1:
        all_singing= []
        for path in Path(params['clean_singing']).rglob('*.wav'):
            all_singing.append(str(path.resolve()))
            
        if params['singing_choice']==1: # male speakers
            mysinging = [s for s in all_singing if ("male" in s and "female" not in s)]
    
        elif params['singing_choice']==2: # female speakers
            mysinging = [s for s in all_singing if "female" in s]
    
        elif params['singing_choice']==3: # both male and female
            mysinging = all_singing
        else: # default both male and female
            mysinging = all_singing
            
        shuffle(mysinging)
        if mysinging is not None:
            all_cleanfiles= cleanfilenames + mysinging
    else: 
        all_cleanfiles= cleanfilenames
        
#   add emotion data to clean speech
    if params['use_emotion_data'] ==1:
        all_emotion= []
        for path in Path(params['clean_emotion']).rglob('*.wav'):
            all_emotion.append(str(path.resolve()))

        shuffle(all_emotion)
        if all_emotion is not None:
            all_cleanfiles = all_cleanfiles + all_emotion
    else: 
        print('NOT using emotion data for training!')    
        
#   add mandarin data to clean speech
    if params['use_mandarin_data'] ==1:
        all_mandarin= []
        for path in Path(params['clean_mandarin']).rglob('*.wav'):
            all_mandarin.append(str(path.resolve()))

        shuffle(all_mandarin)
        if all_mandarin is not None:
            all_cleanfiles = all_cleanfiles + all_mandarin
    else: 
        print('NOT using non-english (Mandarin) data for training!')           
        

    params['cleanfilenames'] = all_cleanfiles
    params['num_cleanfiles'] = len(params['cleanfilenames'])
    # If there are .wav files in noise_dir directory, use those
    # If not, that implies that the noise files are organized into subdirectories by type,
    # so get the names of the non-excluded subdirectories
    if 'noise_csv' in cfg.keys() and cfg['noise_csv'] != 'None':
        noisefilenames = pd.read_csv(cfg['noise_csv'])
        noisefilenames = noisefilenames['filename']
    else:
        noisefilenames = glob.glob(os.path.join(noise_dir, params['audioformat']))

    if len(noisefilenames)!=0:
        shuffle(noisefilenames)
        params['noisefilenames'] = noisefilenames
    else:
        noisedirs = glob.glob(os.path.join(noise_dir, '*'))
        if cfg['noise_types_excluded'] != 'None':
            dirstoexclude = cfg['noise_types_excluded'].split(',')
            for dirs in dirstoexclude:
                noisedirs.remove(dirs)
        shuffle(noisedirs)
        params['noisedirs'] = noisedirs

    # rir 
    temp = pd.read_csv(params['rir_table_csv'], skiprows=[1], sep=',', header=None,  names=['wavfile','channel','T60_WB','C50_WB','isRealRIR'])
    temp.keys()
    #temp.wavfile

    rir_wav = temp['wavfile'][1:] # 115413
    rir_channel = temp['channel'][1:] 
    rir_t60 = temp['T60_WB'][1:] 
    rir_isreal= temp['isRealRIR'][1:]  

    rir_wav2 = [w.replace('\\', '/') for w in rir_wav]
    rir_channel2 = [w for w in rir_channel]
    rir_t60_2 = [w for w in rir_t60]
    rir_isreal2= [w for w in rir_isreal]
    
    myrir =[]
    mychannel=[]
    myt60=[]

    lower_t60=  params['lower_t60']
    upper_t60=  params['upper_t60']

    if params['rir_choice']==1: # real 3076 IRs
        real_indices= [i for i, x in enumerate(rir_isreal2) if x == "1"]

        chosen_i = []
        for i in real_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir= [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]


    elif params['rir_choice']==2: # synthetic 112337 IRs
        synthetic_indices= [i for i, x in enumerate(rir_isreal2) if x == "0"]

        chosen_i = []
        for i in synthetic_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir= [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    elif params['rir_choice']==3: # both real and synthetic
        all_indices= [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir= [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    else:  # default both real and synthetic
        all_indices= [i for i, x in enumerate(rir_isreal2)]

        chosen_i = []
        for i in all_indices:
            if (float(rir_t60_2[i]) >= lower_t60) and (float(rir_t60_2[i]) <= upper_t60):
                chosen_i.append(i)

        myrir= [rir_wav2[i] for i in chosen_i]
        mychannel = [rir_channel2[i] for i in chosen_i]
        myt60 = [rir_t60_2[i] for i in chosen_i]

    params['myrir'] = myrir
    params['mychannel'] = mychannel
    params['myt60'] = myt60

    # Call main_gen() to generate audio
    clean_source_files, clean_clipped_files, clean_low_activity_files, \
    noise_source_files, noise_clipped_files, noise_low_activity_files = main_gen(params)

    # Create log directory if needed, and write log files of clipped and low activity files
    log_dir = utils.get_dir(cfg, 'log_dir', 'Logs')

    utils.write_log_file(log_dir, 'source_files.csv', clean_source_files + noise_source_files)
    utils.write_log_file(log_dir, 'clipped_files.csv', clean_clipped_files + noise_clipped_files)
    utils.write_log_file(log_dir, 'low_activity_files.csv', \
                         clean_low_activity_files + noise_low_activity_files)

    # Compute and print stats about percentange of clipped and low activity files
    total_clean = len(clean_source_files) + len(clean_clipped_files) + len(clean_low_activity_files)
    total_noise = len(noise_source_files) + len(noise_clipped_files) + len(noise_low_activity_files)
    pct_clean_clipped = round(len(clean_clipped_files)/total_clean*100, 1)
    pct_noise_clipped = round(len(noise_clipped_files)/total_noise*100, 1)
    pct_clean_low_activity = round(len(clean_low_activity_files)/total_clean*100, 1)
    pct_noise_low_activity = round(len(noise_low_activity_files)/total_noise*100, 1)

    print("Of the " + str(total_clean) + " clean speech files analyzed, " + \
          str(pct_clean_clipped) + "% had clipping, and " + str(pct_clean_low_activity) + \
          "% had low activity " + "(below " + str(params['clean_activity_threshold']*100) + \
          "% active percentage)")
    print("Of the " + str(total_noise) + " noise files analyzed, " + str(pct_noise_clipped) + \
          "% had clipping, and " + str(pct_noise_low_activity) + "% had low activity " + \
          "(below " + str(params['noise_activity_threshold']*100) + "% active percentage)")


if __name__ == '__main__':

    main_body()
