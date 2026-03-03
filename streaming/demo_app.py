# run  python -m streaming.demo_app  --input_wav /data2/wangchengzhong/challenge/noisy_test/noise_reverb_limit/-5db/file_000_snr-5.wav --cuda_graph

import argparse
import time
from pathlib import Path

from streaming.engine import MPNetStreamingEnhancer


def main():
    parser = argparse.ArgumentParser(description="GPU-accelerated streaming MPNet demo")
    parser.add_argument("--checkpoint_file", default="streaming/streamingdemo_g_00035000")
    parser.add_argument("--config_file", default=None)
    parser.add_argument("--input_wav", required=True)
    parser.add_argument("--output_wav", default="streaming/enh_stream_demo.wav")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--threads", type=int, default=100)
    parser.add_argument("--cuda_graph", action="store_true", help="Enable CUDA Graph replay for per-frame streaming step")
    args = parser.parse_args()

    enhancer = MPNetStreamingEnhancer(
        checkpoint_file=args.checkpoint_file,
        config_file=args.config_file,
        device=args.device,
        num_threads=args.threads,
        use_cuda_graph=args.cuda_graph,
    )

    stat = enhancer.enhance_file(args.input_wav, args.output_wav)


    print(f"Saved: {Path(args.output_wav).resolve()}")


if __name__ == "__main__":
    main()
