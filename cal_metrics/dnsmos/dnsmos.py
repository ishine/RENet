# Usage:
# python dnsmos_local.py -t c:\temp\DNSChallenge4_Blindset -o DNSCh4_Blind.csv -p
#

import argparse
import concurrent.futures
import glob
import os

import librosa
import numpy as np
import numpy.polynomial.polynomial as poly
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from requests import session
from tqdm import tqdm

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01

class ComputeScore:
    def __init__(self, primary_model_path) -> None:
        self.onnx_sess = ort.InferenceSession(primary_model_path, providers=['CUDAExecutionProvider'])

        
    def get_polyfit_val_(self, sig, bak, ovr, is_personalized_MOS):
        if is_personalized_MOS:
            coeffs_ovr = np.array([-0.00533021, 0.005101, 1.18058466, -0.11236046])
            coeffs_sig = np.array([-0.01019296, 0.02751166, 1.19576786, -0.24348726])
            coeffs_bak = np.array([-0.04976499, 0.44276479, -0.1644611, 0.96883132])
        else:
            coeffs_ovr = np.array([-0.06766283, 1.11546468, 0.04602535])
            coeffs_sig = np.array([-0.08397278, 1.22083953, 0.0052439])
            coeffs_bak = np.array([-0.13166888, 1.60915514, -0.39604546])

        # 使用 numpy 的 polyval 来计算多项式的值
        sig_poly = np.polyval(coeffs_sig, sig)
        bak_poly = np.polyval(coeffs_bak, bak)
        ovr_poly = np.polyval(coeffs_ovr, ovr)

        return sig_poly, bak_poly, ovr_poly
    def get_polyfit_val(self, sig, bak, ovr, is_personalized_MOS=False):
        if is_personalized_MOS:
            p_ovr = np.poly1d([-0.00533021,  0.005101  ,  1.18058466, -0.11236046])
            p_sig = np.poly1d([-0.01019296,  0.02751166,  1.19576786, -0.24348726])
            p_bak = np.poly1d([-0.04976499,  0.44276479, -0.1644611 ,  0.96883132])
        else:
            p_ovr = np.poly1d([-0.06766283,  1.11546468,  0.04602535])
            p_sig = np.poly1d([-0.08397278,  1.22083953,  0.0052439 ])
            p_bak = np.poly1d([-0.13166888,  1.60915514, -0.39604546])

        sig_poly = p_sig(sig)
        bak_poly = p_bak(bak)
        ovr_poly = p_ovr(ovr)

        return sig_poly, bak_poly, ovr_poly

    def __call__(self, audio, sampling_rate=16000, is_personalized_MOS=False):
        
        len_samples = int(INPUT_LENGTH*sampling_rate)
        while audio.shape[1] < len_samples:
            audio = np.concatenate((audio, audio),axis=1)
        audio = audio[:, :len_samples]
        input_features = np.array(audio).astype('float32') # [np.newaxis,:]

        oi = {'input_1': input_features}
        # p808_oi = {'input_1': p808_input_features}
        # p808_mos = self.p808_onnx_sess.run(None, p808_oi)[0][0][0]
        outputs = self.onnx_sess.run(None, oi)
        outputs = np.array(outputs)[0]

        mos_sig_raw, mos_bak_raw, mos_ovr_raw = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        mos_sig, mos_bak, mos_ovr = self.get_polyfit_val_(mos_sig_raw, mos_bak_raw, mos_ovr_raw, is_personalized_MOS)


        return mos_sig, mos_bak, mos_ovr

def main(args):
    models = glob.glob(os.path.join(args.testset_dir, "*"))
    audio_clips_list = []
    p808_model_path = os.path.join('DNSMOS', 'model_v8.onnx')

    if args.personalized_MOS:
        primary_model_path = os.path.join('pDNSMOS', 'sig_bak_ovr.onnx')
    else:
        primary_model_path = os.path.join('DNSMOS', 'sig_bak_ovr.onnx')

    compute_score = ComputeScore(primary_model_path, p808_model_path)

    rows = []
    clips = []
    clips = glob.glob(os.path.join(args.testset_dir, "*.wav"))
    is_personalized_eval = args.personalized_MOS
    desired_fs = SAMPLING_RATE
    for m in tqdm(models):
        max_recursion_depth = 10
        audio_path = os.path.join(args.testset_dir, m)
        audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
        while len(audio_clips_list) == 0 and max_recursion_depth > 0:
            audio_path = os.path.join(audio_path, "**")
            audio_clips_list = glob.glob(os.path.join(audio_path, "*.wav"))
            max_recursion_depth -= 1
        clips.extend(audio_clips_list)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_url = {executor.submit(compute_score, clip, desired_fs, is_personalized_eval): clip for clip in clips}
        for future in tqdm(concurrent.futures.as_completed(future_to_url)):
            clip = future_to_url[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (clip, exc))
            else:
                rows.append(data)            

    df = pd.DataFrame(rows)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--testset_dir", default='.', 
                        help='Path to the dir containing audio clips in .wav to be evaluated')
    parser.add_argument('-p', "--personalized_MOS", action='store_true', 
                        help='Flag to indicate if personalized MOS score is needed or regular')
    
    args = parser.parse_args()

    main(args)
