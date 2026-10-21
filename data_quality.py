#!/usr/bin/env python3
"""Data-quality checks for the M-O-N-I-C-O training datasets (CPU, no GPU).

Runs before any training run: validates the JSONL schema, finds duplicates
and near-duplicates, reports length/character stats, and checks identity
coverage (every "who are you?" style question must be represented).

Usage:
    python3 data_quality.py                          # check data/monico_identity.jsonl
    python3 data_quality.py --file other.jsonl
    python3 data_quality.py --strict --report report.json

--strict turns warnings into failures (good for CI). Exit code is 0 when
clean, 1 when errors (or warnings under --strict).
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

DEFAULT_FILE = Path(__file__).parent / "data" / "monico_identity.jsonl"

REQUIRED_FIELDS = {"instruction", "input", "output"}

# Identity questions the model MUST know cold — every one needs coverage.
IDENTITY_MUST_COVER = [
    "who are you", "what's your name", "whats your name", "your name",
    "introduce yourself", "what model are you", "are you Muse",
    "are you chatgpt", "who made you", "what are you",
]
# normalized once (lowercase) so matching is case-insensitive
_IDENTITY_PATTERNS = [q.lower().replace("'", "") for q in IDENTITY_MUST_COVER]

REFUSAL_MARKERS = [
    "i'm sorry, i can't", "i am sorry, i cannot", "as an ai",
    "i cannot help with that", "i'm not able to",
]

MAX_OUTPUT_CHARS = 1200   # identity outputs are short/direct by design
MIN_OUTPUT_CHARS = 1


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def row_hash(row: dict) -> str:
    key = norm(row["instruction"]) + "\x00" + norm(row["output"])
    return hashlib.sha256(key.encode()).hexdigest()


def load_rows(path: Path):
    rows, line_errors = [], []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except json.JSONDecodeError as e:
                line_errors.append(f"line {i}: invalid JSON ({e})")
    return rows, line_errors


def check_schema(rows) -> list[str]:
    errs = []
    for i, r in rows:
        if not isinstance(r, dict):
            errs.append(f"line {i}: row is not a JSON object")
            continue
        missing = REQUIRED_FIELDS - set(r)
        if missing:
            errs.append(f"line {i}: missing fields {sorted(missing)}")
        for k in ("instruction", "input", "output"):
            if k in r and not isinstance(r[k], str):
                errs.append(f"line {i}: field {k!r} is not a string")
        if not r.get("instruction", "").strip():
            errs.append(f"line {i}: empty instruction")
    return errs


def check_duplicates(rows):
    errs, warns = [], []
    seen: dict[str, int] = {}
    for i, r in rows:
        h = row_hash(r)
        if h in seen:
            errs.append(f"line {i}: exact duplicate of line {seen[h]} "
                        f"(instruction: {r['instruction'][:60]!r})")
        else:
            seen[h] = i
    # near-duplicates: same instruction, different output
    by_instr: dict[str, list] = collections.defaultdict(list)
    for i, r in rows:
        by_instr[norm(r["instruction"])].append(i)
    for instr, lines in by_instr.items():
        if len(lines) > 4:
            warns.append(f"instruction {instr[:60]!r} appears {len(lines)}x (lines {lines[:5]}...)")
    return errs, warns


def check_lengths(rows):
    errs, warns = [], []
    lens = []
    for i, r in rows:
        n = len(r.get("output", ""))
        lens.append(n)
        if n < MIN_OUTPUT_CHARS:
            errs.append(f"line {i}: empty output")
        if n > MAX_OUTPUT_CHARS:
            warns.append(f"line {i}: output {n} chars > {MAX_OUTPUT_CHARS} "
                         f"(identity style is short; fine for coding, flagging anyway)")
    return errs, warns, lens


def check_style(rows):
    warns = []
    for i, r in rows:
        out = r.get("output", "").lower()
        for m in REFUSAL_MARKERS:
            if m in out:
                warns.append(f"line {i}: refusal-style phrasing {m!r} in output — "
                             f"identity data should be direct, not lecture-y")
                break
    return warns


def check_identity_coverage(rows):
    errs = []
    covered = set()
    for _, r in rows:
        instr = norm(r["instruction"]).replace("'", "")
        for raw, q in zip(IDENTITY_MUST_COVER, _IDENTITY_PATTERNS):
            if q in instr:
                covered.add(raw)
    for q in IDENTITY_MUST_COVER:
        if q not in covered:
            errs.append(f"identity coverage gap: no row matching {q!r}")
    return errs


def stats(rows):
    lens = [len(r.get("output", "")) for _, r in rows]
    ilens = [len(r.get("instruction", "")) for _, r in rows]
    non_ascii = sum(1 for _, r in rows if any(ord(c) > 127 for c in r.get("output", "")))

    def pct(data, p):
        if not data:
            return 0
        s = sorted(data)
        return s[min(len(s) - 1, int(p * len(s)))]

    return {
        "rows": len(rows),
        "output_chars": {"min": min(lens, default=0), "median": pct(lens, 0.5),
                         "p95": pct(lens, 0.95), "max": max(lens, default=0)},
        "instruction_chars": {"min": min(ilens, default=0), "median": pct(ilens, 0.5),
                              "max": max(ilens, default=0)},
        "rows_with_non_ascii": non_ascii,
    }


def run(path: Path, strict: bool = False):
    errors, warnings = [], []
    rows, line_errors = load_rows(path)
    errors.extend(line_errors)
    if not rows:
        return {"errors": ["no rows loaded"], "warnings": [], "stats": {}}

    errors.extend(check_schema(rows))
    e, w = check_duplicates(rows)
    errors.extend(e)
    warnings.extend(w)
    e, w, _ = check_lengths(rows)
    errors.extend(e)
    warnings.extend(w)
    warnings.extend(check_style(rows))
    errors.extend(check_identity_coverage(rows))

    if strict and warnings:
        errors.extend(f"[strict] {w}" for w in warnings)
        warnings = []

    return {"errors": errors, "warnings": warnings, "stats": stats(rows)}


def main(argv) -> int:
    p = argparse.ArgumentParser(description="Check monico-train dataset quality (CPU)")
    p.add_argument("--file", default=str(DEFAULT_FILE), help="JSONL dataset to check")
    p.add_argument("--strict", action="store_true", help="warnings become failures")
    p.add_argument("--report", metavar="JSON", help="also write a JSON report to this file")
    args = p.parse_args(argv)

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: {path} not found")
        return 1

    result = run(path, strict=args.strict)
    st = result["stats"]
    print(f"checked {st.get('rows', 0)} rows in {path}")
    if st:
        oc = st["output_chars"]
        print(f"output chars: min {oc['min']} / median {oc['median']} / p95 {oc['p95']} / max {oc['max']}")
        print(f"rows with non-ascii: {st['rows_with_non_ascii']}")

    for w in result["warnings"]:
        print(f"WARN: {w}")
    for e in result["errors"]:
        print(f"ERROR: {e}")

    if args.report:
        with open(args.report, "w") as f:
            json.dump(result, f, indent=2)
        print(f"report -> {args.report}")

    ok = not result["errors"]
    print(f"RESULT: {'CLEAN' if ok else 'FAILED'} "
          f"({len(result['errors'])} errors, {len(result['warnings'])} warnings)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
