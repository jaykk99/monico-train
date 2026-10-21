#!/usr/bin/env python3
"""Build monico_train_colab.ipynb via nbformat. Run: python3 build_notebook.py
Validates the notebook with nbformat.validate and syntax-checks every code cell.
"""
import ast
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}

def md(src):
    nb.cells.append(new_markdown_cell(src))

def code(src):
    nb.cells.append(new_code_cell(src))

# ------------------------------------------------------------------ 0 intro
md("""# M-O-N-I-C-O — Custom Uncensored Model Training (QLoRA, free Colab)

Fine-tunes a **real custom model** on a free Colab T4 GPU. **Cost: $0.**

**What you get:** a LoRA adapter + optional GGUF file trained on coding instructions,
uncensored chat, and M-O-N-I-C-O identity data (short, direct, no-lecture style).

**How to run:** Colab → Runtime → Change runtime type → **T4 GPU** → Runtime → **Run all**.
Go make coffee. ~1–2 hours total.

**Honest notes (read me):**
- Free Colab gives you ~12h GPU per day and ~12GB RAM. This notebook is sized to fit:
  7B model in 4-bit + ~12k training examples ≈ 30–60 min for 1 epoch on a T4.
- If Colab kills your session (usage limits), just re-run — the adapter saves to
  Google Drive, so nothing is lost.
- The GGUF export merges the full 7B weights and needs lots of RAM. It **may OOM**
  on free-tier RAM. If it does, the LoRA adapter in Drive is still the complete
  trained result — merge it later on any machine with 16GB+ RAM.
- This trains a LoRA fine-tune of Dolphin-2.9.2-Qwen2-7B (already uncensored).
  You're teaching it M-O-N-I-C-O's identity + sharpening coding — not training
  from scratch (that costs millions).
""")

# ------------------------------------------------------------------ 1 gpu
code('''import torch
assert torch.cuda.is_available(), "NO GPU DETECTED — set Runtime > Change runtime type > T4 GPU, then re-run."
name = torch.cuda.get_device_name(0)
gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU: {name} ({gb:.1f} GB)")
assert gb > 10, "Need 10GB+ VRAM (T4 has ~15GB). Pick a T4 runtime."
print("GPU check passed.")
''')

# ------------------------------------------------------------------ 2 install
md("### 1. Install (takes ~3–5 min, one time per session)")
code('''%%capture
import torch
# Unsloth's Colab installer picks the right build for your GPU.
# PINNED to the 2026-09-19 commit so re-runs don't drift with unsloth git HEAD.
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@768d644036a9"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
print("Install done. If the NEXT cell throws import errors, do Runtime > Restart session and skip this cell on re-run.")
''')

# ------------------------------------------------------------------ 3 drive
md("### 2. Mount Google Drive (everything saves here — keyless, no tokens needed)")
code('''from google.colab import drive
drive.mount('/content/drive')
import os
OUT = "/content/drive/MyDrive/monico-train"
os.makedirs(OUT, exist_ok=True)
print("Saving to:", OUT)
''')

# ------------------------------------------------------------------ 4 model
md("""### 3. Load the base model (4-bit QLoRA)

Base: **Dolphin-2.9.2-Qwen2-7B** — uncensored, strong at general chat + coding.
Loaded in 4-bit so a 7B model fits on a free T4.""")
code('''from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="cognitivecomputations/Dolphin-2.9.2-Qwen2-7B",
    max_seq_length=2048,
    dtype=None,          # auto: bf16 on T4
    load_in_4bit=True,   # 4-bit QLoRA — the whole trick that makes this free
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",  # 30% less VRAM
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)
print("Model loaded + LoRA attached.")
''')

# ------------------------------------------------------------------ 5 data
md("""### 4. Load the datasets (all public, keyless)

- **Code:** Magicoder-OSS-Instruct-75K → 6,000 sampled (`problem`/`solution`)
- **Uncensored chat:** QuixiAI/dolphin (flan1m-alpaca-uncensored) → 6,000 sampled
- **M-O-N-I-C-O identity:** 200 examples from this repo (cloned below)""")
code('''!git clone -q https://github.com/jaykk99/monico-train /content/monico-train 2>/dev/null || echo "repo already cloned"

from datasets import load_dataset

code_ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train") \\
    .shuffle(seed=42).select(range(6000))
print("code rows:", len(code_ds), "| fields:", code_ds.column_names)

chat_ds = load_dataset("QuixiAI/dolphin", "flan1m-alpaca-uncensored", split="train") \\
    .shuffle(seed=42).select(range(6000))
print("chat rows:", len(chat_ds), "| fields:", chat_ds.column_names)

id_ds = load_dataset("json", data_files="/content/monico-train/data/monico_identity.jsonl", split="train")
print("identity rows:", len(id_ds), "| fields:", id_ds.column_names)
''')

# ------------------------------------------------------------------ 6 format
md("### 5. Format everything into one training set")
code('''from datasets import concatenate_datasets

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS = tokenizer.eos_token
def fmt(text): return {"text": text + EOS}

code_fmt = code_ds.map(lambda ex: fmt(alpaca_prompt.format(ex["problem"], "", ex["solution"])),
                       remove_columns=code_ds.column_names)
chat_fmt = chat_ds.map(lambda ex: fmt(alpaca_prompt.format(ex["instruction"], ex["input"] or "", ex["output"])),
                       remove_columns=chat_ds.column_names)
id_fmt = id_ds.map(lambda ex: fmt(alpaca_prompt.format(ex["instruction"], ex["input"], ex["output"])),
                   remove_columns=id_ds.column_names)

train_ds = concatenate_datasets([code_fmt, chat_fmt, id_fmt]).shuffle(seed=3407)
print("TOTAL training rows:", len(train_ds))
print("--- sample ---")
print(train_ds[0]["text"][:400])
''')

# ------------------------------------------------------------------ 7 trainer
md("""### 6. Training setup

1 epoch ≈ 1,525 steps ≈ 30–60 min on a T4. Bump `num_train_epochs` to 2 if you
want it sharper and have the free-tier hours.""")
code('''from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=2048,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch = 8
        warmup_steps=20,
        num_train_epochs=1,              # 1 = safe on free tier; 2 = sharper
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)
print("Trainer ready.")
''')

# ------------------------------------------------------------------ 8 train
md("### 7. TRAIN — this is the long cell (~30–60 min). Don't close the tab.")
code('''stats = trainer.train()
print("Training done. Loss:", stats.training_loss)
''')

# ------------------------------------------------------------------ 9 test
md("### 8. Sanity check — talk to your new model")
code('''from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Who are you?

### Input:


### Response:
"""

inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=80, use_cache=True)
print(tokenizer.batch_decode(out)[0].split("### Response:")[-1].strip())
''')

# ------------------------------------------------------------------ 10 save
md("### 9. Save the adapter to Google Drive (this is your trained model)")
code('''model.save_pretrained(f"{OUT}/monico-lora")
tokenizer.save_pretrained(f"{OUT}/monico-lora")
print("Saved to", f"{OUT}/monico-lora")
!ls -lh {OUT}/monico-lora | head
''')

# ------------------------------------------------------------------ 11 gguf notes
md("""### 10. GGUF export (for Ollama / llama.cpp on your Chromebook)

**Honest warning:** this merges the full 7B weights and quantizes them — it needs
a lot of system RAM and **may OOM on free Colab** (~12GB RAM). If it crashes, your
trained adapter in Drive is 100% fine — merge it later on any machine with 16GB+ RAM
using the `merge.py` script in the repo, or re-run this cell on a high-RAM runtime.
""")
code('''# Best-effort GGUF export. If this OOMs, the adapter above is still your trained model.
model.save_pretrained_gguf(f"{OUT}/gguf", tokenizer, quantization_method="q4_k_m")
print("GGUF saved to", f"{OUT}/gguf")
''')

# ------------------------------------------------------------------ 12 hf
md("""### 11. (Optional) Push to HuggingFace

Only if you want the model on the Hub. Needs a free HF token — skip if you want
to stay keyless; Drive already has everything.""")
code('''# --- OPTIONAL: uncomment to push to HuggingFace ---
# from huggingface_hub import login
# login("hf_YOUR_TOKEN_HERE")  # paste a write token from huggingface.co/settings/tokens
# model.push_to_hub_gguf("jaykk99/monico-7b", tokenizer, quantization_method="q4_k_m")
# print("Pushed to huggingface.co/jaykk99/monico-7b")
print("Skipped (optional). Uncomment the lines above if you want the Hub upload.")
''')

md("""## Done. What you have now

- `MyDrive/monico-train/monico-lora/` — the trained LoRA adapter (the actual trained weights)
- `MyDrive/monico-train/gguf/` — Q4_K_M GGUF (if the export survived RAM limits)

**Run it on your Chromebook:** install Ollama, then
`ollama create monico -f Modelfile` with `FROM ./monico-q4_k_m.gguf`.
Or point llama.cpp / LM Studio / Jan at the GGUF directly.

**If the GGUF step OOM'd:** run `python3 merge.py` from this repo on any machine
with 16GB+ RAM — it merges the adapter and spits out the GGUF locally.
""")

# ------------------------------------------------------------- validate
import argparse as _ap
from pathlib import Path as _Path
_ap_p = _ap.ArgumentParser(description="Regenerate monico_train_colab.ipynb")
_ap_p.add_argument("--out", default=str(_Path(__file__).parent / "monico_train_colab.ipynb"),
                   help="notebook output path (default: repo root)")
path = _ap_p.parse_args().out
nbformat.write(nb, path)
nbformat.validate(nb)
print(f"nbformat validation passed: {len(nb.cells)} cells")

n_code = 0
for i, cell in enumerate(nb.cells):
    if cell.cell_type == "code":
        lines = []
        for ln in cell.source.split("\n"):
            s = ln.strip()
            if s.startswith("%%") or s.startswith("!") or (s.startswith("%") and not s.startswith("%%")):
                continue  # cell magics / shell escapes aren't Python
            lines.append(ln)
        src = "\n".join(lines)
        try:
            ast.parse(src)
        except SyntaxError as e:
            raise SystemExit(f"SYNTAX ERROR in code cell {i}: {e}")
        n_code += 1
print(f"syntax check passed: {n_code} code cells")
print("wrote", path)
