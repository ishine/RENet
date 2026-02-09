# applicable to both dns21 and dns20 training datalist generalization 
import os
import glob
import re

def extract_fileid_from_noisy_filename(filename):
    """Extract fileid from noisy filename like 'book_11346_chp_0030_reader_00812_56_XHdrRVey_w4-YR2AHUO0ISQ-8ubTow4S_6w_snr10_fileid_118096.wav'"""
    # Extract the fileid part from the filename
    match = re.search(r'fileid_(\d+)\.wav$', filename)
    if match:
        return f"clean_fileid_{match.group(1)}.wav"
    return None

def generate_dns_training_file(noisy_dir, clean_dir, output_file):
    """Generate training file for DNS dataset"""
    
    # Get all noisy files
    noisy_files = glob.glob(os.path.join(noisy_dir, "*.wav"))
    print(f"Found {len(noisy_files)} noisy files")
    
    # Create mapping from fileid to noisy file path
    fileid_to_noisy = {}
    for noisy_file in noisy_files:
        filename = os.path.basename(noisy_file)
        fileid = extract_fileid_from_noisy_filename(filename)
        if fileid:
            fileid_to_noisy[fileid] = noisy_file
        else:
            print(f"Warning: Could not extract fileid from {filename}")
    
    print(f"Successfully mapped {len(fileid_to_noisy)} noisy files")
    
    # Check which clean files exist
    clean_files = glob.glob(os.path.join(clean_dir, "*.wav"))
    clean_filenames = {os.path.basename(f) for f in clean_files}
    print(f"Found {len(clean_files)} clean files")
    
    # Generate training file
    with open(output_file, 'w') as f:
        for fileid, noisy_file in fileid_to_noisy.items():
            if fileid in clean_filenames:
                # Format: fileid|noisy_file_path
                f.write(f"{fileid}|{noisy_file}\n")
            else:
                print(f"Warning: Clean file {fileid} not found")
    
    print(f"Generated training file: {output_file}")
    print(f"Total pairs: {len([line for line in open(output_file)])}")

if __name__ == "__main__":
    # "/data/home/wangchengzhong/dtb/DNS-Challenge/noisy", "/data/home/wangchengzhong/dtb/DNS-Challenge/clean" "DNS-Challenge/training.txt" for DNS training
    # "/data2/wangchengzhong/challenge/noisy", "/data2/wangchengzhong/challenge/clean" "WHAMR/training.txt" for universal training
    noisy_dir = "/root/autodl-tmp/challenge/noisy"
    clean_dir = "/root/autodl-tmp/challenge/clean"
    output_file = "WHAMR/training.txt"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    generate_dns_training_file(noisy_dir, clean_dir, output_file) 