#!/usr/bin/env python3
"""Fallback: merge the trained LoRA adapter into the base model and export GGUF.

Use this on any machine with 16GB+ RAM if the Colab GGUF-export cell OOM'd
(free Colab only has ~12GB RAM). Your trained adapter from Google Drive is the
input — nothing is lost.

Usage:
    pip install "unsloth @ git+https://github.com/unslothai/unsloth.git" torch
    python3 merge.py --adapter /path/to/monico-lora --out ./monico-merged --gguf

Notes:
- Needs the base model downloadable from HuggingFace (keyless, public).
- Merging is CPU/RAM heavy (~14GB for 7B fp16). GPU speeds it up but isn't required.
- --gguf also writes monico-q4_k_m.gguf next to the merged model.
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser(description="Merge M-O-N-I-C-O LoRA adapter + export GGUF")
    p.add_argument("--adapter", required=True, help="Path to monico-lora adapter dir (from Drive)")
    p.add_argument("--base", default="cognitivecomputations/Dolphin-2.9.2-Qwen2-7B",
                   help="Base model (must match training)")
    p.add_argument("--out", default="./monico-merged", help="Output dir for merged 16-bit model")
    p.add_argument("--gguf", action="store_true", help="Also export Q4_K_M GGUF")
    args = p.parse_args()

    from unsloth import FastLanguageModel

    print(f"Loading adapter {args.adapter} (base: {args.base}) ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter,   # Unsloth reads adapter_config.json -> finds base
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,       # full precision for a clean merge
    )

    print(f"Merging to {args.out} ...")
    model.save_pretrained_merged(args.out, tokenizer, save_method="merged_16bit")
    print("Merged model saved.")

    if args.gguf:
        print("Exporting Q4_K_M GGUF (this takes a while) ...")
        model.save_pretrained_gguf(args.out + "-gguf", tokenizer, quantization_method="q4_k_m")
        print("GGUF saved to", args.out + "-gguf")


if __name__ == "__main__":
    main()
