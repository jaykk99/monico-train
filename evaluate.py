#!/usr/bin/env python3
"""CPU eval harness for the M-O-N-I-C-O fine-tune (no GPU needed).

Scores model outputs against eval/golden.jsonl: each golden row has an
instruction, required/forbidden substrings (case-insensitive), a max word
count, and a category. This runs on a laptop — it scores a SAVED file of
model responses, so you generate responses once (Colab, Ollama, anywhere)
and grade them here.

Two modes:
  1. Grade a responses file (JSONL: {"instruction": ..., "response": ...}):
       python3 evaluate.py --responses my_run.jsonl
  2. Self-check mode (no model): matches each golden instruction against the
     training data and reports which golden prompts are/aren't covered —
     i.e. "will the model even have seen this?"
       python3 evaluate.py --coverage --data data/monico_identity.jsonl

Golden response generation helper:
       python3 evaluate.py --template --out my_run.jsonl
  writes a template file you fill with model outputs (or pipe from a script).

Exit 0 if every row passes, 1 otherwise. --json prints machine-readable results.
"""
import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
GOLDEN = HERE / "eval" / "golden.jsonl"
IDENTITY_DATA = HERE / "data" / "monico_identity.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise SystemExit(f"{path} line {i}: bad JSON ({e})")
    return rows


def word_count(text: str) -> int:
    return len(text.split())


def grade_row(golden: dict, response: str) -> dict:
    """Score one (instruction, response) pair against the golden expectations."""
    resp_low = response.lower()
    failures = []

    for need in golden.get("must_contain", []):
        variants = [need.lower()] if isinstance(need, str) else [v.lower() for v in need]
        if not any(v in resp_low for v in variants):
            failures.append(f"missing required {need!r}")

    for banned in golden.get("must_not_contain", []):
        if banned.lower() in resp_low:
            failures.append(f"contains forbidden {banned!r}")

    max_w = golden.get("max_words")
    if max_w and word_count(response) > max_w:
        failures.append(f"{word_count(response)} words > max {max_w}")

    if not response.strip():
        failures.append("empty response")

    return {
        "instruction": golden["instruction"],
        "category": golden.get("category", "?"),
        "passed": not failures,
        "failures": failures,
    }


def mode_grade(responses_path: Path, as_json: bool) -> int:
    golden = load_jsonl(GOLDEN)
    resp_rows = load_jsonl(responses_path)
    by_instr = {r["instruction"]: r.get("response", "") for r in resp_rows}

    results = []
    missing = 0
    for g in golden:
        instr = g["instruction"]
        if instr not in by_instr:
            missing += 1
            results.append({"instruction": instr, "category": g.get("category", "?"),
                            "passed": False, "failures": ["no response provided"]})
        else:
            results.append(grade_row(g, by_instr[instr]))

    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    passed = sum(1 for r in results if r["passed"])
    if as_json:
        print(json.dumps({"passed": passed, "total": len(results), "missing": missing,
                          "results": results}, indent=2))
    else:
        for cat, rs in sorted(by_cat.items()):
            ok = sum(1 for r in rs if r["passed"])
            print(f"[{cat}] {ok}/{len(rs)} passed")
            for r in rs:
                if not r["passed"]:
                    print(f"  FAIL {r['instruction'][:60]!r}")
                    for f in r["failures"]:
                        print(f"       - {f}")
        print(f"\nTOTAL: {passed}/{len(results)} passed"
              + (f" ({missing} missing responses)" if missing else ""))

    return 0 if (passed == len(results) and not missing) else 1


def _tokset(s: str) -> set:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def mode_coverage(data_path: Path, as_json: bool) -> int:
    """For each golden prompt, find the best-matching training row (token overlap)."""
    golden = load_jsonl(GOLDEN)
    data = load_jsonl(data_path)
    results = []
    for g in golden:
        gtok = _tokset(g["instruction"])
        best, best_score = None, 0.0
        for d in data:
            dtok = _tokset(d.get("instruction", ""))
            if not gtok or not dtok:
                continue
            score = len(gtok & dtok) / len(gtok | dtok)
            if score > best_score:
                best, best_score = d, score
        results.append({
            "instruction": g["instruction"],
            "category": g.get("category", "?"),
            "best_overlap": round(best_score, 2),
            "covered": best_score >= 0.5,
            "best_match": (best or {}).get("instruction", "")[:80],
        })

    covered = sum(1 for r in results if r["covered"])
    if as_json:
        print(json.dumps({"covered": covered, "total": len(results), "results": results}, indent=2))
    else:
        for r in results:
            mark = "OK " if r["covered"] else "GAP"
            print(f"[{mark}] ({r['category']}) overlap={r['best_overlap']:.2f} "
                  f"{r['instruction'][:55]!r}")
            if not r["covered"]:
                print(f"       closest training row: {r['best_match']!r}")
        print(f"\nCOVERAGE: {covered}/{len(results)} golden prompts have a close training row")
        print("(GAPs are expected for coding/general rows — those come from the "
              "Magicoder/dolphin datasets, not the identity file.)")
    return 0


def mode_template(out_path: Path) -> int:
    golden = load_jsonl(GOLDEN)
    with out_path.open("w", encoding="utf-8") as f:
        for g in golden:
            f.write(json.dumps({"instruction": g["instruction"], "response": ""}) + "\n")
    print(f"wrote {len(golden)}-row template -> {out_path}")
    print("fill in 'response' with your model's outputs, then run:")
    print(f"  python3 evaluate.py --responses {out_path}")
    return 0


def main(argv) -> int:
    p = argparse.ArgumentParser(description="CPU eval harness for M-O-N-I-C-O (no GPU)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--responses", metavar="JSONL", help="grade this responses file")
    g.add_argument("--coverage", action="store_true",
                   help="check golden prompts against training data")
    g.add_argument("--template", action="store_true",
                   help="write an empty responses template")
    p.add_argument("--data", default=str(IDENTITY_DATA),
                   help="training JSONL for --coverage")
    p.add_argument("--out", default="eval/responses_template.jsonl",
                   help="output path for --template")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)

    if not GOLDEN.exists():
        print(f"ERROR: golden file missing: {GOLDEN}")
        return 1

    if args.template:
        return mode_template(Path(args.out))
    if args.coverage:
        return mode_coverage(Path(args.data), args.json)
    return mode_grade(Path(args.responses), args.json)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
