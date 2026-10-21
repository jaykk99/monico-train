# Data quality & eval for monico-train (all CPU, no GPU)

Training data bugs are the most expensive kind: you only find them 45 minutes
into a Colab run. These tools check everything on a laptop *before* training.

## Quick check before any training run

```bash
python3 config.py                  # validate hyperparameters
python3 data_quality.py --strict   # validate datasets (fails CI-style on warnings)
python3 -m unittest discover -s tests   # full test suite
```

## Tools

### config.py — hyperparameter validation
Single source of truth for the training config (matches the notebook values).
Rejects typos (negative lr, r=0, empty target modules), warns when the batch
size won't fit a free T4, and checks the data mix is non-empty with enough
identity rows for the M-O-N-I-C-O voice to stick.

```bash
python3 config.py --dump my_config.json   # start from defaults
# edit my_config.json ...
python3 config.py --check my_config.json
```

### data_quality.py — dataset checks
Runs over `data/monico_identity.jsonl` (or `--file`):

- JSONL validity + required schema (`instruction`/`input`/`output`, all strings)
- exact duplicates (SHA-256 of normalized instruction+output)
- near-duplicate warnings (same instruction reused 5+ times)
- output length outliers (empty; >1200 chars flagged for review)
- refusal-style phrasing in outputs ("as an AI", "I'm sorry, I can't"...)
- **identity coverage**: every "who are you?" style question must appear
- length/character stats (min/median/p95/max, non-ASCII count)

```bash
python3 data_quality.py --strict --report quality_report.json
```

Exit 0 = clean, 1 = errors (or warnings under `--strict`).

### evaluate.py — CPU eval harness (no model needed to run the tooling)
Scores *saved* model responses against `eval/golden.jsonl` (20 prompts:
identity, direct/uncensored, coding, general, business, style). Each golden row
declares required substrings, forbidden substrings, and a max word count.

```bash
# 1. generate a template, 2. fill it with your model's outputs, 3. grade
python3 evaluate.py --template --out eval/my_run.jsonl
# ... fill in "response" fields (from Colab, Ollama, wherever) ...
python3 evaluate.py --responses eval/my_run.jsonl
```

Coverage mode (no model at all): checks which golden prompts have a close
training row — tells you what the identity data alone can teach:

```bash
python3 evaluate.py --coverage
```

Coding/general GAPs are expected there — those categories come from the
Magicoder and dolphin datasets, not the identity file.

## Regenerating the notebook

`build_notebook.py` regenerates `monico_train_colab.ipynb` from Python
(validates with nbformat + AST-checks every code cell):

```bash
python3 build_notebook.py              # writes to repo root (default)
python3 build_notebook.py --out /tmp/nb.ipynb
```

The test suite verifies the committed notebook is always a fresh builder
output — if you edit the notebook by hand, re-run the builder and commit.
