"""Tests for data_quality.py — dataset checks. CPU only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import data_quality as dq

GOOD = [
    {"instruction": "Who are you?", "input": "", "output": "I'm M-O-N-I-C-O — Jay's right hand."},
    {"instruction": "What's your name?", "input": "", "output": "M-O-N-I-C-O."},
    {"instruction": "Introduce yourself.", "input": "", "output": "M-O-N-I-C-O, built for Jay."},
    {"instruction": "What model are you?", "input": "", "output": "M-O-N-I-C-O on a Dolphin 7B base."},
    {"instruction": "Are you Muse?", "input": "", "output": "Nope, M-O-N-I-C-O."},
    {"instruction": "Are you ChatGPT?", "input": "", "output": "No, M-O-N-I-C-O."},
    {"instruction": "Who made you?", "input": "", "output": "Jay."},
    {"instruction": "What are you?", "input": "", "output": "M-O-N-I-C-O, an AI assistant."},
]


def write_jsonl(rows):
    f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
    for r in rows:
        f.write(json.dumps(r) + "\n")
    f.close()
    return Path(f.name)


class TestSchema(unittest.TestCase):
    def test_good_rows_clean(self):
        rows = [(i + 1, r) for i, r in enumerate(GOOD)]
        errs = dq.check_schema(rows)
        self.assertEqual(errs, [])

    def test_missing_field(self):
        rows = [(1, {"instruction": "x", "input": ""})]
        self.assertTrue(any("missing" in e for e in dq.check_schema(rows)))

    def test_empty_instruction(self):
        rows = [(1, {"instruction": " ", "input": "", "output": "y"})]
        self.assertTrue(any("empty instruction" in e for e in dq.check_schema(rows)))


class TestDuplicates(unittest.TestCase):
    def test_exact_duplicate_flagged(self):
        rows = [(1, GOOD[0]), (2, dict(GOOD[0]))]
        errs, _ = dq.check_duplicates(rows)
        self.assertTrue(any("duplicate" in e for e in errs))

    def test_instruction_reused_5x_warns(self):
        rows = [(i + 1, {"instruction": "Who are you?", "input": "",
                         "output": f"answer variant {i}"}) for i in range(5)]
        _, warns = dq.check_duplicates(rows)
        self.assertTrue(any("5x" in w for w in warns))


class TestLengths(unittest.TestCase):
    def test_empty_output_error(self):
        rows = [(1, {"instruction": "q", "input": "", "output": ""})]
        errs, _, _ = dq.check_lengths(rows)
        self.assertTrue(any("empty output" in e for e in errs))

    def test_giant_output_warns(self):
        rows = [(1, {"instruction": "q", "input": "", "output": "x" * 5000})]
        _, warns, _ = dq.check_lengths(rows)
        self.assertTrue(any("chars" in w for w in warns))


class TestStyle(unittest.TestCase):
    def test_refusal_phrasing_flagged(self):
        rows = [(1, {"instruction": "q", "input": "",
                     "output": "I'm sorry, I can't help with that."})]
        warns = dq.check_style(rows)
        self.assertEqual(len(warns), 1)

    def test_direct_answer_clean(self):
        rows = [(1, GOOD[0])]
        self.assertEqual(dq.check_style(rows), [])


class TestIdentityCoverage(unittest.TestCase):
    def test_full_coverage_passes(self):
        rows = [(i + 1, r) for i, r in enumerate(GOOD)]
        self.assertEqual(dq.check_identity_coverage(rows), [])

    def test_gap_flagged(self):
        rows = [(1, GOOD[0])]
        errs = dq.check_identity_coverage(rows)
        self.assertTrue(errs)  # only "who are you" covered


class TestRunEndToEnd(unittest.TestCase):
    def test_run_on_good_file(self):
        p = write_jsonl(GOOD)
        res = dq.run(p)
        self.assertEqual(res["stats"]["rows"], 8)
        self.assertEqual(res["errors"], [], f"errors: {res['errors']}")

    def test_run_on_real_dataset_is_clean(self):
        real = Path(__file__).resolve().parent.parent / "data" / "monico_identity.jsonl"
        self.assertTrue(real.exists(), "real dataset must exist")
        res = dq.run(real)
        self.assertEqual(res["stats"]["rows"], 200)
        self.assertEqual(res["errors"], [], f"real dataset errors: {res['errors'][:5]}")

    def test_strict_promotes_warnings(self):
        rows = [(1, GOOD[0])]  # coverage gaps -> errors anyway; use refusal warn
        p = write_jsonl([{"instruction": "Who are you?", "input": "",
                          "output": "As an AI, I am M-O-N-I-C-O."}])
        res = dq.run(p, strict=True)
        self.assertTrue(any("strict" in e for e in res["errors"]))
        self.assertEqual(res["warnings"], [])

    def test_bad_json_line(self):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        f.write('{"instruction": "x", "input": "", "output": "y"}\nnot json\n')
        f.close()
        res = dq.run(Path(f.name))
        self.assertTrue(any("invalid JSON" in e for e in res["errors"]))


class TestNorm(unittest.TestCase):
    def test_norm_collapses_whitespace_and_case(self):
        self.assertEqual(dq.norm("  Who   ARE\nYou? "), "who are you?")


if __name__ == "__main__":
    unittest.main()
