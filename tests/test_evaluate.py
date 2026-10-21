"""Tests for evaluate.py — CPU grading of model responses. CPU only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import evaluate as ev


class TestGradeRow(unittest.TestCase):
    def base_golden(self, **kw):
        g = {"instruction": "Who are you?", "must_contain": ["m-o-n-i-c-o"],
             "must_not_contain": ["i'm muse"], "max_words": 60, "category": "identity"}
        g.update(kw)
        return g

    def test_pass(self):
        r = ev.grade_row(self.base_golden(), "I'm M-O-N-I-C-O — Jay's right hand.")
        self.assertTrue(r["passed"])
        self.assertEqual(r["failures"], [])

    def test_missing_required(self):
        r = ev.grade_row(self.base_golden(), "I'm an assistant.")
        self.assertFalse(r["passed"])
        self.assertTrue(any("missing required" in f for f in r["failures"]))

    def test_forbidden_present(self):
        r = ev.grade_row(self.base_golden(), "I'm Muse, actually M-O-N-I-C-O.")
        self.assertFalse(r["passed"])
        self.assertTrue(any("forbidden" in f for f in r["failures"]))

    def test_too_long(self):
        r = ev.grade_row(self.base_golden(), "M-O-N-I-C-O " + "word " * 100)
        self.assertFalse(r["passed"])
        self.assertTrue(any("words > max" in f for f in r["failures"]))

    def test_empty_response(self):
        r = ev.grade_row(self.base_golden(), "   ")
        self.assertFalse(r["passed"])
        self.assertTrue(any("empty" in f for f in r["failures"]))

    def test_case_insensitive(self):
        r = ev.grade_row(self.base_golden(), "I AM M-O-N-I-C-O.")
        self.assertTrue(r["passed"])

    def test_must_contain_list_variants(self):
        g = self.base_golden(must_contain=[["m-o-n-i-c-o", "monico"]])
        r = ev.grade_row(g, "Monico at your service.")
        self.assertTrue(r["passed"])


class TestModeGrade(unittest.TestCase):
    def write_responses(self, pairs):
        f = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)
        for instr, resp in pairs:
            f.write(json.dumps({"instruction": instr, "response": resp}) + "\n")
        f.close()
        return Path(f.name)

    def test_missing_responses_fail(self):
        p = self.write_responses([])
        # golden has 20 rows; empty file -> all missing -> nonzero
        self.assertEqual(ev.main(["--responses", str(p)]), 1)

    def test_golden_file_valid(self):
        golden = ev.load_jsonl(ev.GOLDEN)
        self.assertGreaterEqual(len(golden), 15)
        for g in golden:
            self.assertIn("instruction", g)
            self.assertIn("category", g)
            self.assertIsInstance(g.get("must_contain", []), list)

    def test_template_then_grade_roundtrip(self):
        out = Path(tempfile.mkdtemp()) / "tpl.jsonl"
        self.assertEqual(ev.main(["--template", "--out", str(out)]), 0)
        rows = ev.load_jsonl(out)
        golden = ev.load_jsonl(ev.GOLDEN)
        self.assertEqual(len(rows), len(golden))
        # fill with canned good answers that satisfy the golden constraints
        canned = {
            "identity": "I'm M-O-N-I-C-O, Jay's right hand. Short and direct.",
            "direct": "Straight answer: yes, here's the deal, no lecture.",
            "coding": "def reverse(s):\n    return s[::-1]\n\nUse json.load for files.",
            "general": "Ottawa. 36. Application Programming Interface.",
            "business": "204-972-0325. 30% deposit before the job, balance on completion.",
            "style": "Write the one thing down tonight. Do it first tomorrow.",
        }
        filled = []
        for r, g in zip(rows, golden):
            ans = canned.get(g["category"], "M-O-N-I-C-O answer.")
            # patch answers that need specific tokens
            low = g["instruction"].lower()
            if "capital of canada" in low:
                ans = "Ottawa."
            elif "15%" in low:
                ans = "36."
            elif "api stand" in low:
                ans = "Application Programming Interface."
            elif "reverses a string" in low:
                ans = "def reverse(s):\n    return s[::-1]"
            elif "json file" in low:
                ans = "Use json.load to read it."
            elif "undo my last commit" in low:
                ans = "git reset --soft HEAD~1"
            elif "phone number" in low:
                ans = "204-972-0325."
            elif "payment terms" in low:
                ans = "30% deposit, balance on completion."
            elif "who made you" in low:
                ans = "Jay made me. I'm M-O-N-I-C-O."
            elif "torrent site" in low or "torrenting" in low:
                ans = "1337x is the usual answer. Straight talk, no lecture."
            elif "scrape" in low:
                ans = "Scraping public data is generally fine."
            filled.append({"instruction": r["instruction"], "response": ans})
        with out.open("w") as f:
            for r in filled:
                f.write(json.dumps(r) + "\n")
        rc = ev.main(["--responses", str(out)])
        self.assertEqual(rc, 0, "canned good answers should all pass")


class TestModeCoverage(unittest.TestCase):
    def test_coverage_runs_on_identity_data(self):
        rc = ev.main(["--coverage"])
        self.assertEqual(rc, 0)  # coverage never fails; it reports

    def test_coverage_json_shape(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            ev.main(["--coverage", "--json"])
        data = json.loads(buf.getvalue())
        self.assertIn("covered", data)
        self.assertIn("results", data)


if __name__ == "__main__":
    unittest.main()
