import sys
import os
import glob
import subprocess
import re
import tempfile
import tablib
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
from argparse import ArgumentParser
from tqdm import tqdm
from pesq import pesq
from pystoi import stoi
from cal_metrics.compute_metrics import compute_metrics
# Use torch.multiprocessing for better CUDA handling
import torch.multiprocessing as mp

# Add paths provided in the original script
sys.path.append("cal_metrics/UTMOS_demo")
# sys.path.append("dnsmos") # Uncomment if needed

from cal_metrics.dnsmos.dnsmos_p808_local import ComputeScore_
from cal_metrics.UTMOS_demo.score import Score
import logging



# STFT Settings (Matches BAPEN paper: 32ms window, 8ms hop for 16kHz)
N_FFT = 400
HOP_SIZE = 100
WIN_SIZE = 400

ALL_METRICS = (
    "PESQ",
    "STOI",
    "SISNR",
    "CSIG",
    "CBAK",
    "COVL",
    "DNSMOS",
    "UTMOS",
    "PD",
    "WOPD",
)

METRICS_TO_CALC = set(ALL_METRICS)

# --- HELPER FUNCTIONS FOR PHASE METRICS ---

def mag_pha_stft(y, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    # Ensure input is (Batch, Time)
    if y.dim() == 1:
        y = y.unsqueeze(0)

    hann_window = torch.hann_window(win_size).to(y.device)
    # return_complex=True is preferred in newer PyTorch, but we stick to the user's snippet style adapted for compatibility
    stft_spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                           center=center, pad_mode='reflect', normalized=False, return_complex=True)
    
    stft_spec = torch.view_as_real(stft_spec)
    mag = torch.sqrt(stft_spec.pow(2).sum(-1) + 1e-9)
    pha = torch.atan2(stft_spec[:, :, :, 1] + 1e-10, stft_spec[:, :, :, 0] + 1e-5)
    
    # Magnitude Compression
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)

    return mag, pha, com

def anti_wrapping_function(y_true, y_pred):
    """
    Computes the anti-wrapped difference between two phase tensors.
    f_AW(t) = |t - 2*pi * round(t / 2*pi)|
    Result is in radians, in the range [0, pi].
    """
    diff = y_pred - y_true
    return torch.abs(diff - 2 * np.pi * torch.round(diff / (2 * np.pi)))

class PhaseMetricsCalculator(nn.Module):
    def __init__(self):
        super(PhaseMetricsCalculator, self).__init__()
        # Define kernels (9 directions for WOPD)
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
        
        kernels = torch.from_numpy(kernels_np).unsqueeze(1) # (9, 1, 3, 3)
        self.register_buffer('kernels', kernels)

    def forward(self, true_phase, pred_phase, true_magnitude):
        """
        Calculates PD and WOPD.
        
        1. PD (Phase Distance):
           - Defined in MP-SENet (Lu et al., 2025) and Choi et al. (2019) [cite: 1229].
           - Definition: Average angle difference weighted by target magnitude.
           - Unit: Degrees (0 to 180)[cite: 1287].
           
        2. WOPD (Weighted Omni-directional Phase Difference):
           - Defined in BAPEN (Eq. 16).
           - Definition: Magnitude-weighted mean of anti-wrapped differences over 9 directions.
           - Unit: Radians (kept consistent with loss formulation).
        """
        
        # --- 1. Calculate PD (Degrees, Weighted) ---
        # Calculate instantaneous anti-wrapped difference (radians)
        aw_diff_rad = anti_wrapping_function(true_phase, pred_phase)
        
        # Convert to Degrees [cite: 1231]
        aw_diff_deg = aw_diff_rad * (180.0 / np.pi)
        
        # Calculate Magnitude-Weighted Mean
        # PD = sum(Mag * Diff_deg) / sum(Mag)
        pd_score = torch.sum(true_magnitude * aw_diff_deg) / (torch.sum(true_magnitude) + 1e-9)

        
        # --- 2. Calculate WOPD (Radians, Weighted, 9 Directions) ---
        # Ensure 4D shape for conv2d
        if true_phase.dim() == 3: true_phase = true_phase.unsqueeze(1)
        if pred_phase.dim() == 3: pred_phase = pred_phase.unsqueeze(1)

        # Convolve to get differences in 9 directions
        delta_p_true = F.conv2d(true_phase, self.kernels, padding='same')
        delta_p_pred = F.conv2d(pred_phase, self.kernels, padding='same')

        # Calculate anti-wrapped difference for all directions
        # Note: anti_wrapping_function handles broadcasting if shapes match
        loss_map_wopd = anti_wrapping_function(delta_p_true, delta_p_pred)
        
        # Weight by magnitude (Broadcast magnitude to 9 channels)
        mag_expanded = true_magnitude.unsqueeze(1) 
        
        # WOPD = sum(Mag * LossMap) / sum(Mag * 9)
        wopd_score = torch.sum(mag_expanded * loss_map_wopd) / (torch.sum(mag_expanded) * 9.0 + 1e-9)

        return pd_score.item(), wopd_score.item()

# --- WORKER INITIALIZATION ---

def init_worker(metrics_to_calc):
    """Initializer for each worker process. Runs ONCE per worker."""
    global dnsmos_model, utmos_model, phase_metric_calc, METRICS_TO_CALC
    METRICS_TO_CALC = set(metrics_to_calc)
    
    # 1. DNSMOS
    if "DNSMOS" in METRICS_TO_CALC:
        try:
            # Update path if necessary
            dnsmos_model = ComputeScore_("/data/home/wangchengzhong/gdse/sgmse/sgmse/util/dnsmos/model_v8.onnx")
        except Exception as e:
            print(f"Worker {os.getpid()} failed to load DNSMOS: {e}")
            dnsmos_model = None
    else:
        dnsmos_model = None
        
    # 2. UTMOS
    if "UTMOS" in METRICS_TO_CALC:
        try:
            utmos_model = Score()
        except Exception as e:
            print(f"Worker {os.getpid()} failed to load UTMOS: {e}")
            utmos_model = None
    else:
        utmos_model = None

    # 3. Phase Metrics Calculator
    if "PD" in METRICS_TO_CALC or "WOPD" in METRICS_TO_CALC:
        try:
            phase_metric_calc = PhaseMetricsCalculator()
            # Ensure it's on CPU to avoid CUDA contention in workers unless explicitly managed
            phase_metric_calc.cpu() 
        except Exception as e:
            print(f"Worker {os.getpid()} failed to load PhaseCalculator: {e}")
            phase_metric_calc = None
    else:
        phase_metric_calc = None

# --- METRIC FUNCTIONS ---

def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    S_target = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    e_noise = out_sig - S_target
    ratio = np.sum(S_target ** 2) / (np.sum(e_noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr



def evaluate_file(args):
    """
    Worker function. Uses global models loaded in init_worker.
    """
    clean_path, enhance_path, fs = args
    try:
        clean_speech, sr_c = sf.read(clean_path)
        enhanced_speech, sr_e = sf.read(enhance_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return None

    target_sr = 16000
    if sr_c != target_sr:
        clean_speech = librosa.resample(clean_speech, orig_sr=sr_c, target_sr=target_sr)
    if sr_e != target_sr:
        enhanced_speech = librosa.resample(enhanced_speech, orig_sr=sr_e, target_sr=target_sr)

    lengths = min(len(clean_speech), len(enhanced_speech))
    clean_speech = clean_speech[:lengths]
    enhanced_speech = enhanced_speech[:lengths]

    metrics = {}

    # --- Standard Metrics ---
    # DNSMOS
    if "DNSMOS" in METRICS_TO_CALC:
        try:
            metrics["DNSMOS"] = dnsmos_model(enhanced_speech) if dnsmos_model else 0.0
        except Exception:
            metrics["DNSMOS"] = 0.0

    # SI-SNR
    if "SISNR" in METRICS_TO_CALC:
        metrics["SISNR"] = cal_SISNR(clean_speech, enhanced_speech)

    # Composite (PESQ/STOI/CSIG/CBAK/COVL)
    if METRICS_TO_CALC.intersection({"PESQ", "STOI", "CSIG", "CBAK", "COVL"}):
        try:
            pesq_score, sig_score, bak_score, ovl_score, _, stoi_score = compute_metrics(
                clean_speech, enhanced_speech, 16000, 0
            )
        except Exception:
            pesq_score, sig_score, bak_score, ovl_score, stoi_score = 0, 0, 0, 0, 0

        if "PESQ" in METRICS_TO_CALC:
            metrics["PESQ"] = pesq_score
        if "STOI" in METRICS_TO_CALC:
            metrics["STOI"] = stoi_score
        if "CSIG" in METRICS_TO_CALC:
            metrics["CSIG"] = sig_score
        if "CBAK" in METRICS_TO_CALC:
            metrics["CBAK"] = bak_score
        if "COVL" in METRICS_TO_CALC:
            metrics["COVL"] = ovl_score

    # UTMOS
    if "UTMOS" in METRICS_TO_CALC:
        try:
            if utmos_model:
                dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                metrics["UTMOS"] = utmos_model.score(
                    torch.from_numpy(enhanced_speech.astype(np.float32)).to(device=dev)
                )[0].item()
            else:
                metrics["UTMOS"] = 0.0
        except Exception:
            metrics["UTMOS"] = 0.0

    # --- Phase Metrics (PD & WOPD) ---
    if ("PD" in METRICS_TO_CALC or "WOPD" in METRICS_TO_CALC) and phase_metric_calc:
        try:
            # Prepare tensors (CPU is sufficient for evaluation usually)
            clean_tensor = torch.from_numpy(clean_speech).float()
            enhanced_tensor = torch.from_numpy(enhanced_speech).float()

            # Compute STFT
            mag_clean, pha_clean, _ = mag_pha_stft(clean_tensor, N_FFT, HOP_SIZE, WIN_SIZE)
            _, pha_enh, _ = mag_pha_stft(enhanced_tensor, N_FFT, HOP_SIZE, WIN_SIZE)
            
            # Compute PD/WOPD
            # Note: Phase metrics usually compare Enhanced Phase to Clean Phase
            # and weight by Clean Magnitude (Target Magnitude).
            with torch.no_grad():
                pd_score, wopd_score = phase_metric_calc(pha_clean, pha_enh, mag_clean)
            if "PD" in METRICS_TO_CALC:
                metrics["PD"] = pd_score
            if "WOPD" in METRICS_TO_CALC:
                metrics["WOPD"] = wopd_score
        except Exception as e:
            print(f"Phase metric error: {e}")
            if "PD" in METRICS_TO_CALC:
                metrics["PD"] = 0.0
            if "WOPD" in METRICS_TO_CALC:
                metrics["WOPD"] = 0.0

    return (os.path.basename(enhance_path), metrics)

def process_single_folder(clean_dir, enhance_dir, file_list, sample_rate, save_path, pool, metrics_to_calc):
    headers = ("audio_names", *metrics_to_calc)
    
    paths = [(os.path.join(clean_dir, f), os.path.join(enhance_dir, f), sample_rate) for f in file_list]

    results = list(tqdm(pool.imap(evaluate_file, paths), total=len(paths), desc=f"Eval {os.path.basename(enhance_dir)}", leave=False))

    metrics_seq = [r for r in results if r is not None]

    if not metrics_seq:
        return None

    metrics_seq = sorted(metrics_seq, key=lambda x: x[0])

    rows = []
    for name, metric_dict in metrics_seq:
        row = [name]
        for m in metrics_to_calc:
            row.append(metric_dict.get(m, 0.0))
        rows.append(tuple(row))

    cols = list(zip(*rows))
    means = []
    stds = []

    for i in range(1, len(headers)):
        data = np.array(cols[i])
        means.append(np.mean(data))
        stds.append(np.std(data))

    rows.append(('MEAN', *means))
    rows.append(('STD', *stds))

    data = tablib.Dataset(*rows, headers=headers)
    with open(save_path, "wb") as f:
        f.write(data.export("csv").encode('utf-8'))
    
    return means 

def evaluation_hierarchical(clean_root, enhance_root, excel_root_name, sample_rate, metrics_to_calc, num_workers=6):
    if not os.path.exists(enhance_root):
        print(f"Enhanced directory not found: {enhance_root}")
        return

    summary_headers = ("Case", *metrics_to_calc)
    summary_rows = []

    print(f"Initializing Pool with {num_workers} workers (loading models once)...")
    
    ctx = mp.get_context('spawn')
    
    with ctx.Pool(processes=num_workers, initializer=init_worker, initargs=(metrics_to_calc,)) as pool:
            # case_name = 'noise_reverb_limit'
        for case_name in sorted(os.listdir(enhance_root)):
            case_path = os.path.join(enhance_root, case_name)
            clean_case_path = os.path.join(clean_root, case_name)
            
            if not os.path.isdir(case_path):
                continue

            wav_files = sorted(glob.glob(os.path.join(case_path, '*.wav')))
            wav_names = [os.path.basename(w) for w in wav_files]

            # Flat Case
            if len(wav_names) > 0:
                print(f"Processing Flat Case: {case_name}")
                csv_name = f"res_{case_name}.csv"
                means = process_single_folder(clean_case_path, case_path, wav_names, sample_rate, csv_name, pool, metrics_to_calc)
                if means:
                    summary_rows.append((case_name, *means))
            
            # Nested Case
            else:
                sub_dirs = sorted(os.listdir(case_path))
                sub_case_means = []
                print(f"Processing Nested Case: {case_name} ({len(sub_dirs)} subfolders)")

                for sub_name in sub_dirs:
                    sub_path = os.path.join(case_path, sub_name)
                    clean_sub_path = os.path.join(clean_case_path, sub_name)
                    
                    if not os.path.isdir(sub_path):
                        continue
                    
                    sub_wavs = sorted(glob.glob(os.path.join(sub_path, '*.wav')))
                    sub_wav_names = [os.path.basename(w) for w in sub_wavs]
                    
                    if len(sub_wav_names) == 0:
                        continue

                    csv_name = f"res_{case_name}_{sub_name}.csv"
                    means = process_single_folder(clean_sub_path, sub_path, sub_wav_names, sample_rate, csv_name, pool, metrics_to_calc)
                    
                    if means:

                        sub_case_means.append(means)

                if sub_case_means:
                    avg_means = np.mean(np.array(sub_case_means), axis=0)
                    summary_rows.append((case_name, *avg_means))

    print(f"Writing Summary: {excel_root_name}")
    data = tablib.Dataset(*summary_rows, headers=summary_headers)
    with open(excel_root_name, "wb") as f:
        f.write(data.export("csv").encode('utf-8'))

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = ArgumentParser()
    # /data2/wangchengzhong/challenge/remixed_vbd_test/clean
    # /data2/wangchengzhong/challenge/clean_test
    parser.add_argument("--clean_dir", type=str, default='/data2/wangchengzhong/challenge/clean_test') # replace this with your clean_test dir
    parser.add_argument("--enhanced_dir", type=str)
    parser.add_argument("--excel_name", type=str, required=True)
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help=(
            "Comma-separated metric list or 'all'. "
            "Choices: PESQ,STOI,SISNR,CSIG,CBAK,COVL,DNSMOS,UTMOS,PD,WOPD"
        ),
    )
    args = parser.parse_args()
    
    SAMPLE_RATE = 16000 
    
    if args.metrics.strip().lower() == "all":
        metrics_to_calc = list(ALL_METRICS)
    else:
        metrics_to_calc = [m.strip().upper() for m in args.metrics.split(",") if m.strip()]
        invalid = [m for m in metrics_to_calc if m not in ALL_METRICS]
        if invalid:
            raise ValueError(f"Unknown metrics: {invalid}. Valid: {', '.join(ALL_METRICS)}")

    evaluation_hierarchical(
        args.clean_dir,
        args.enhanced_dir,
        args.excel_name,
        SAMPLE_RATE,
        metrics_to_calc,
        num_workers=6,
    )