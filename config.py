#!/usr/bin/env python3
"""Training configuration with validation.

Single source of truth for the hyperparameters used by the notebook
(build_notebook.py), merge.py, and the data pipeline. Validating here
catches bad configs on a laptop instead of 30 minutes into a Colab GPU run.

Usage:
    python3 config.py                      # print defaults + validate
    python3 config.py --check my.json      # validate a JSON config file
    python3 config.py --dump defaults.json # write defaults to a file

All constraints are loose but sane: they reject typos (negative lr, r=0)
while still letting you experiment.
"""
import json
import sys
from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    # model
    base_model: str = "cognitivecomputations/Dolphin-2.9.2-Qwen2-7B"
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    # LoRA
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj")
    use_rslora: bool = False
    random_state: int = 3407
    # data
    code_rows: int = 6000
    chat_rows: int = 6000
    identity_rows: int = 200
    dataset_seed: int = 42
    # training
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    warmup_steps: int = 20
    logging_steps: int = 10
    optim: str = "adamw_8bit"
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    seed: int = 3407
    output_dir: str = "outputs"


ERRORS: list[str] = []


def _err(msg: str) -> None:
    ERRORS.append(msg)


def _check_range(name: str, val: float, lo: float, hi: float) -> None:
    if not (lo <= val <= hi):
        _err(f"{name}={val} out of sane range [{lo}, {hi}]")


def validate(cfg: TrainConfig) -> list[str]:
    """Return a list of problems; empty means the config is fine."""
    global ERRORS
    ERRORS = []

    # model
    if not cfg.base_model or "/" not in cfg.base_model:
        _err(f"base_model={cfg.base_model!r} doesn't look like a HF repo id")
    _check_range("max_seq_length", cfg.max_seq_length, 512, 32768)

    # LoRA
    _check_range("lora_r", cfg.lora_r, 1, 256)
    _check_range("lora_alpha", cfg.lora_alpha, 1, 512)
    _check_range("lora_dropout", cfg.lora_dropout, 0.0, 0.9)
    if not cfg.target_modules:
        _err("target_modules is empty — nothing would be trained")
    for m in cfg.target_modules:
        if not m.endswith("_proj"):
            _err(f"target_modules entry {m!r} doesn't look like an attention/MLP proj")
    if cfg.lora_r > 64 and cfg.use_rslora:
        _err("lora_r > 64 with use_rslora=True is unusual — did you mean r=16?")

    # data
    for name in ("code_rows", "chat_rows", "identity_rows"):
        v = getattr(cfg, name)
        _check_range(name, v, 0, 200_000)
    total = cfg.code_rows + cfg.chat_rows + cfg.identity_rows
    if total == 0:
        _err("total training rows = 0 — nothing to train on")
    if cfg.identity_rows < 50:
        _err(f"identity_rows={cfg.identity_rows} < 50 — identity may not stick")

    # training
    eff_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
    _check_range("per_device_train_batch_size", cfg.per_device_train_batch_size, 1, 64)
    _check_range("gradient_accumulation_steps", cfg.gradient_accumulation_steps, 1, 512)
    _check_range("effective_batch", eff_batch, 1, 512)
    _check_range("num_train_epochs", cfg.num_train_epochs, 0.1, 10.0)
    _check_range("learning_rate", cfg.learning_rate, 1e-7, 1e-2)
    _check_range("warmup_steps", cfg.warmup_steps, 0, 1000)
    _check_range("weight_decay", cfg.weight_decay, 0.0, 1.0)
    if cfg.optim not in ("adamw_8bit", "adamw_torch", "adamw_torch_fused",
                         "sgd", "adafactor", "paged_adamw_8bit"):
        _err(f"optim={cfg.optim!r} is not a known transformers optimizer")
    if cfg.lr_scheduler_type not in ("linear", "cosine", "cosine_with_restarts",
                                     "polynomial", "constant", "constant_with_warmup"):
        _err(f"lr_scheduler_type={cfg.lr_scheduler_type!r} is not a known scheduler")
    if cfg.output_dir.startswith("/"):
        _err(f"output_dir={cfg.output_dir!r} is absolute — relative keeps Colab sessions portable")

    # VRAM sanity for a free T4 (15GB)
    # Rough: 7B in 4-bit ~5GB + LoRA/optimizer overhead ~ batch * seqlen.
    est_gb = 5.0 + 0.15 * cfg.per_device_train_batch_size * (cfg.max_seq_length / 2048)
    if est_gb > 13:
        _err(f"estimated VRAM ~{est_gb:.1f}GB exceeds a free T4 (15GB) — lower batch size or seq length")

    return ERRORS


def from_dict(d: dict) -> TrainConfig:
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    kw = {}
    for f in fields(TrainConfig):
        if f.name in d:
            kw[f.name] = d[f.name]
    if isinstance(kw.get("target_modules"), list):
        kw["target_modules"] = tuple(kw["target_modules"])
    return TrainConfig(**kw)


def main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser(description="Validate monico-train hyperparameter config")
    p.add_argument("--check", metavar="JSON", help="validate a JSON config file")
    p.add_argument("--dump", metavar="JSON", help="write default config to a file and exit")
    args = p.parse_args(argv)

    if args.dump:
        with open(args.dump, "w") as f:
            json.dump({**asdict(TrainConfig()), "target_modules": list(TrainConfig.target_modules)},
                      f, indent=2)
        print(f"wrote defaults -> {args.dump}")
        return 0

    if args.check:
        with open(args.check) as f:
            cfg = from_dict(json.load(f))
        src = args.check
    else:
        cfg = TrainConfig()
        src = "defaults"

    problems = validate(cfg)
    if problems:
        print(f"config '{src}' INVALID ({len(problems)} problem(s)):")
        for pr in problems:
            print(f"  - {pr}")
        return 1
    print(f"config '{src}' OK — effective batch {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}, "
          f"{cfg.code_rows + cfg.chat_rows + cfg.identity_rows} rows, {cfg.num_train_epochs} epoch(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
