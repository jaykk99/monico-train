# monico-train — Train your own M-O-N-I-C-O model

A one-click Google Colab notebook that fine-tunes a **real custom uncensored model**
with QLoRA. Free tier, $0, no API keys needed.

## What it trains

- **Base:** `cognitivecomputations/Dolphin-2.9.2-Qwen2-7B` (uncensored, strong general + coding), loaded in 4-bit
- **Method:** QLoRA via Unsloth (r=16, alpha=16, 1 epoch)
- **Data (~12,200 examples):**
  - 6,000 coding instructions (Magicoder-OSS-Instruct-75K)
  - 6,000 uncensored chat examples (QuixiAI/dolphin)
  - 200 M-O-N-I-C-O identity examples ([`data/monico_identity.jsonl`](data/monico_identity.jsonl)) — short, casual, direct, no-lecture style
- **Output:** LoRA adapter + Q4_K_M GGUF saved to your Google Drive

## Run it (Jay's steps)

1. Open the notebook in Colab:
   **https://colab.research.google.com/github/jaykk99/monico-train/blob/master/monico_train_colab.ipynb**
2. **Runtime → Change runtime type → T4 GPU** → Save
3. **Runtime → Run all**
4. Click "Connect" on the Google Drive prompt when it appears
5. Wait ~1–2 hours. Done.

**Cost:** $0 (Colab free tier). **Time:** ~30–60 min training + setup.

## What you get

In Google Drive → `MyDrive/monico-train/`:
- `monico-lora/` — the trained adapter (the actual trained weights)
- `gguf/` — `monico-q4_k_m.gguf` (if the export fit in free-tier RAM)

## Run it on your Chromebook

Install Ollama, then:
```
ollama create monico -f Modelfile
ollama run monico
```
With a `Modelfile` containing: `FROM ./monico-q4_k_m.gguf`

Or open the GGUF directly in LM Studio / Jan / llama.cpp.

## If the GGUF export runs out of memory

Free Colab has ~12GB RAM and the merge needs ~14GB. Your trained adapter is
still complete — run this on any machine with 16GB+ RAM:
```
pip install "unsloth @ git+https://github.com/unslothai/unsloth.git@768d644036a9" "torch>=2.6"   # unsloth pinned 2026-09-19; torch 2.6 is unsloth's oldest supported
python3 merge.py --adapter /path/to/monico-lora --out ./monico-merged --gguf
```

## Files

| File | What |
|---|---|
| `monico_train_colab.ipynb` | The one-click training notebook |
| `generate_identity_data.py` | Generates the 200 M-O-N-I-C-O identity examples (seeded, reproducible) |
| `data/monico_identity.jsonl` | The generated identity dataset |
| `merge.py` | Fallback: merge adapter + export GGUF off-Colab |
| `build_notebook.py` | Regenerates the notebook (validates JSON + syntax) |

## Honest limits

- This fine-tunes an existing uncensored 7B — it teaches M-O-N-I-C-O's identity
  and sharpens coding. Training a foundation model from scratch costs millions.
- No GPU run has been done yet — the notebook is validated (syntax, data pipeline,
  dataset fields) but the actual training happens when you hit Run all in Colab.
- Free Colab can kill long sessions; the adapter saves to Drive so re-runs are cheap.
