import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
sys.path.append("..")
import os
import time
import argparse
import json
import torch
import glob
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from dns_dataset import DNSDataset, get_dns_dataset_filelist
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet, pesq_score, phase_losses
from models.mpd_and_metricd import MetricDiscriminator, batch_pesq,  MultiPeriodDiscriminator, \
     feature_loss, generator_loss, discriminator_loss
from utils import scan_checkpoint, load_checkpoint, save_checkpoint

def get_hierarchical_loaders(noisy_root, clean_root, h, device):
    """
    Scans the noisy_root directory for validation cases.
    It handles two types of folder structures:
    1. Flat: noisy_root/case_name/*.wav (e.g., only_reverb)
    2. Nested: noisy_root/case_name/sub_case/*.wav (e.g., only_noise/-5db, only_bandlimit/2khz)
    """
    loaders = {}
    if not os.path.exists(noisy_root):
        print(f"Warning: Test noisy directory {noisy_root} not found.")
        return loaders

    # Iterate over the top-level cases (e.g., only_reverb, only_noise, only_bandlimit)
    for case_name in sorted(os.listdir(noisy_root)):
        case_path = os.path.join(noisy_root, case_name)
        if not os.path.isdir(case_path):
            continue

        # CHECK 1: Look for wav files directly in this folder (Depth 1)
        wav_files_depth1 = sorted(glob.glob(os.path.join(case_path, '*.wav')))
        
        # Take only the first 20%. Uses max(1, ...) to ensure at least 1 file is tested if dir is not empty.
        if len(wav_files_depth1) > 0:
            limit = max(5, int(len(wav_files_depth1) * 0.1))
            wav_files_depth1 = wav_files_depth1[:limit]

        if len(wav_files_depth1) > 0:
            # Create loader for this flat case
            pairs = []
            for w in wav_files_depth1:
                rel_path = os.path.relpath(w, noisy_root)
                pairs.append((rel_path, w))
            
            ds = DNSDataset(pairs, clean_root, h.segment_size, h.sampling_rate,
                            split=False, shuffle=False, n_cache_reuse=0, device=device)
            loader = DataLoader(ds, num_workers=1, shuffle=False, 
                                sampler=None, batch_size=1, pin_memory=True, drop_last=False)
            
            loaders[case_name] = loader
            print(f"Loaded validation case '{case_name}': {len(pairs)} files (Subset 20%)")
        
        else:
            # CHECK 2: Look for sub-directories (Depth 2)
            sub_dirs = sorted(os.listdir(case_path))
            for sub_name in sub_dirs:
                sub_path = os.path.join(case_path, sub_name)
                if not os.path.isdir(sub_path):
                    continue
                
                wav_files_depth2 = sorted(glob.glob(os.path.join(sub_path, '*.wav')))

                # Take only the first 20% for nested subcases
                if len(wav_files_depth2) > 0:
                    limit = max(5, int(len(wav_files_depth2) * 0.1))
                    wav_files_depth2 = wav_files_depth2[:limit]

                if len(wav_files_depth2) > 0:
                    pairs = []
                    for w in wav_files_depth2:
                        rel_path = os.path.relpath(w, noisy_root)
                        pairs.append((rel_path, w))
                    
                    ds = DNSDataset(pairs, clean_root, h.segment_size, h.sampling_rate,
                                    split=False, shuffle=False, n_cache_reuse=0, device=device)
                    loader = DataLoader(ds, num_workers=1, shuffle=False, 
                                        sampler=None, batch_size=1, pin_memory=True, drop_last=False)
                    
                    key_name = f"{case_name}_{sub_name}"
                    loaders[key_name] = loader
                    print(f"Loaded validation case '{key_name}': {len(pairs)} files (Subset 20%)")

    return loaders


def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = MPNet(h).to(device)
    discriminator = MetricDiscriminator().to(device)
    mpd = MultiPeriodDiscriminator().to(device)

    if rank == 0:
        print(generator)
        num_params = 0
        for p in generator.parameters():
            num_params += p.numel()
        print('Total Parameters: {:.3f}M'.format(num_params/1e6))
        os.makedirs(a.checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(a.checkpoint_path, 'logs'), exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
    
    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])

        discriminator.load_state_dict(state_dict_do['discriminator'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']
    
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(discriminator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_m = torch.optim.AdamW(mpd.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])
        optim_m.load_state_dict(state_dict_do['optim_m'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_m = torch.optim.lr_scheduler.ExponentialLR(optim_m, gamma=h.lr_decay, last_epoch=last_epoch)

    training_pairs, _ = get_dns_dataset_filelist(a.input_training_file, a.input_validation_file)

    trainset = DNSDataset(training_pairs, a.input_clean_wavs_dir, h.segment_size, h.sampling_rate,
                          split=True, n_cache_reuse=0, shuffle=False if h.num_gpus > 1 else True, device=device)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)
    
    val_loaders_dict = {}
    if rank == 0:
        print("Initializing hierarchical validation loaders...")
        val_loaders_dict = get_hierarchical_loaders(a.test_noisy_dir, a.test_clean_dir, h, device)
        print(f"Found {len(val_loaders_dict)} subcases for validation: {list(val_loaders_dict.keys())}")
        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq = 0

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            if rank == 0:
                start_b = time.time()
            clean_audio, noisy_audio = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))
            one_labels = torch.ones(h.batch_size).to(device, non_blocking=True)

            clean_mag, clean_pha, clean_com = mag_pha_stft(clean_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            noisy_mag, noisy_pha, noisy_com = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)
            audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            mag_g_hat, pha_g_hat, com_g_hat = mag_pha_stft(audio_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

            # Discriminator
            optim_m.zero_grad()
            audio_df_r, audio_df_g, _, _ = mpd(clean_audio.unsqueeze(1), audio_g.detach().unsqueeze(1))
            loss_disc_f, _, _ = discriminator_loss(audio_df_r, audio_df_g)

            
            loss_disc_mpd = loss_disc_f
            loss_disc_mpd.backward()
            torch.nn.utils.clip_grad_norm_(parameters=mpd.parameters(), max_norm=10, norm_type=2)

            optim_m.step()
            
            # Calculate PESQ for Discriminator
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g)

            # Discriminator Step
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g_hat.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            
            loss_disc_metric = loss_disc_r + loss_disc_g
            loss_disc_metric.backward()
            optim_d.step()

            # Generator Step
            optim_g.zero_grad()
            loss_mag = F.mse_loss(clean_mag, mag_g)
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g)
            loss_pha = loss_ip + loss_gd + loss_iaf
            loss_com = F.mse_loss(clean_com, com_g) * 2
            loss_stft = F.mse_loss(com_g, com_g_hat) * 2
            loss_time = F.l1_loss(clean_audio, audio_g)
            metric_g = discriminator(clean_mag, mag_g_hat)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            # mpd loss
            _, audio_df_g, fmap_f_r, fmap_f_g = mpd(clean_audio.unsqueeze(1), audio_g.unsqueeze(1))
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_gen_f, _ = generator_loss(audio_df_g)
            loss_mpd = loss_fm_f + loss_gen_f
            
            loss_gen_all = loss_mag * 0.9 + loss_pha * 0.3  + loss_com * 0.1 + loss_stft * 0.1 + loss_metric * 0.05 + loss_time * 0.2  + loss_mpd * 0.05
            loss_gen_all.backward()

            optim_g.step()

            if rank == 0:
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mpd_error = loss_mpd.detach().mean().item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        pha_error = loss_pha.item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        stft_error = F.mse_loss(com_g, com_g_hat).item()
                        print('Steps : {:d}, Gen Loss: {:4.3f}, DiscPESQ Loss: {:4.3f}, DiscMPD Loss: {:4.3f}, Metric loss: {:4.3f}, MPD loss: {:4.3f}, Magnitude Loss : {:4.3f}, Phase Loss : {:4.3f}, Complex Loss : {:4.3f}, Time Loss : {:4.3f}, STFT Loss : {:4.3f}, s/b : {:4.3f}'.
                            format(steps, loss_gen_all, loss_disc_metric, loss_disc_mpd, metric_error, mpd_error, mag_error, pha_error, com_error, time_error, stft_error, time.time() - start_b))

                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'discriminator': (discriminator.module if h.num_gpus > 1 else discriminator).state_dict(),
                                     'mpd': (mpd.module if h.num_gpus > 1 else mpd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'optim_m': optim_m.state_dict(), 'steps': steps, 
                                     'epoch': epoch})

                if steps % a.summary_interval == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/DiscriminatorMetric Loss", loss_disc_metric, steps)
                    sw.add_scalar("Training/DiscriminatorMPD Loss", loss_disc_mpd, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistency Loss", stft_error, steps)

                # --- VALIDATION LOOP ---
                if steps % a.validation_interval == 0 and steps != 0 and len(val_loaders_dict) > 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    
                    global_pesq_sum = 0
                    total_subcases = 0
                    
                    print(f"Starting validation at step {steps}...")
                    
                    with torch.no_grad():
                        # Iterate through each subcase (e.g., noisy_-5db, noisy_0db, ...)
                        for case_name, v_loader in val_loaders_dict.items():
                            audios_r, audios_g = [], []
                            
                            for batch in v_loader:
                                clean_audio, noisy_audio = batch
                                clean_audio = clean_audio.to(device, non_blocking=True)
                                noisy_audio = noisy_audio.to(device, non_blocking=True)

                                noisy_mag, noisy_pha, _ = mag_pha_stft(noisy_audio, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
                                mag_g, pha_g, _ = generator(noisy_mag, noisy_pha)
                                audio_g = mag_pha_istft(mag_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)

                                audios_r += torch.split(clean_audio, 1, dim=0)
                                audios_g += torch.split(audio_g, 1, dim=0)

                            # Compute PESQ for this specific subcase
                            case_pesq = pesq_score(audios_r, audios_g, h).item()
                            
                            # Log to TensorBoard
                            sw.add_scalar(f"Validation/PESQ_{case_name}", case_pesq, steps)
                            print(f"  {case_name}: PESQ = {case_pesq:.3f}")
                            
                            global_pesq_sum += case_pesq
                            total_subcases += 1

                    # Compute Global PESQ (average of all subcases) for checkpointing
                    global_avg_pesq = global_pesq_sum / total_subcases if total_subcases > 0 else 0
                    sw.add_scalar("Validation/Global_PESQ", global_avg_pesq, steps)
                    print(f"Global Average PESQ: {global_avg_pesq:.3f}")

                    # Save Best Model based on Global Average PESQ
                    if epoch >= a.best_checkpoint_start_epoch:
                        if global_avg_pesq > best_pesq:
                            best_pesq = global_avg_pesq
                            best_checkpoint_path = "{}/g_best".format(a.checkpoint_path)
                            save_checkpoint(best_checkpoint_path,
                                            {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                            print(f"New best model saved with PESQ: {best_pesq:.3f}")

                    generator.train()
            
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        scheduler_m.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_clean_wavs_dir', default='/root/autodl-tmp/challenge/clean')
    
    parser.add_argument('--test_noisy_dir', default='/root/autodl-tmp/challenge/noisy_test')
    parser.add_argument('--test_clean_dir', default='/root/autodl-tmp/challenge/clean_test')

    parser.add_argument('--input_training_file', default='DNS21_filelist/training.txt')
    parser.add_argument('--input_validation_file', default='VBD_filelist/test.txt') # This arg is kept but effectively unused for validation logic now
    parser.add_argument('--checkpoint_path', default='cp_model_universe')
    parser.add_argument('--config', default='config_universal.json')
    parser.add_argument('--training_epochs', default=400, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=5000, type=int)
    parser.add_argument('--best_checkpoint_start_epoch', default=0, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config_universel.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)

if __name__ == '__main__':
    main()