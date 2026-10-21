"""Pipeline tests: build_notebook.py, merge.py, generate_identity_data.py.

- build_notebook: regenerates the notebook to a temp dir, validates it.
- merge.py: argparse surface (never imports unsloth — no GPU deps in tests).
- generate_identity_data: 200 rows, exact schema, seeded reproducibility,
  byte-identical to the committed data/monico_identity.jsonl.
CPU only; no network; no GPU.
"""
import ast
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class TestBuildNotebook(unittest.TestCase):
    def test_regenerates_and_validates(self):
        nbformat = __import__("nbformat")
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "nb.ipynb")
            rc = subprocess.run([sys.executable, str(ROOT / "build_notebook.py"),
                                 "--out", out],
                                capture_output=True, text=True, cwd=str(ROOT))
            self.assertEqual(rc.returncode, 0, f"build failed:\n{rc.stderr[-2000:]}")
            nb = nbformat.read(out, as_version=4)
            nbformat.validate(nb)
            code_cells = [c for c in nb.cells if c.cell_type == "code"]
            self.assertGreaterEqual(len(code_cells), 10)
            src = "\n".join(c.source for c in nb.cells)
            for needle in ("FastLanguageModel", "SFTTrainer", "save_pretrained_gguf",
                           "cognitivecomputations/Dolphin-2.9.2-Qwen2-7B", "768d644036a9"):
                self.assertIn(needle, src, f"notebook missing {needle!r}")

    def test_code_cells_parse(self):
        nbformat = __import__("nbformat")
        nb = nbformat.read(str(ROOT / "monico_train_colab.ipynb"), as_version=4)
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            lines = [ln for ln in cell.source.split("\n")
                     if not ln.strip().startswith(("%%", "!"))
                     and not (ln.strip().startswith("%") and not ln.strip().startswith("%%"))]
            try:
                ast.parse("\n".join(lines))
            except SyntaxError as e:
                self.fail(f"committed notebook code cell {i}: {e}")


class TestMergeArgs(unittest.TestCase):
    def test_help_lists_options(self):
        # simpler: run --help which exercises argparse without heavy imports
        rc = subprocess.run([sys.executable, str(ROOT / "merge.py"), "--help"],
                            capture_output=True, text=True)
        self.assertEqual(rc.returncode, 0)
        self.assertIn("--adapter", rc.stdout)
        self.assertIn("--base", rc.stdout)
        self.assertIn("--gguf", rc.stdout)

    def test_missing_adapter_errors(self):
        rc = subprocess.run([sys.executable, str(ROOT / "merge.py")],
                            capture_output=True, text=True)
        self.assertNotEqual(rc.returncode, 0)
        self.assertIn("--adapter", rc.stderr)


class TestIdentityData(unittest.TestCase):
    def test_200_rows_exact_schema(self):
        path = ROOT / "data" / "monico_identity.jsonl"
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(rows), 200)
        for r in rows:
            self.assertEqual(set(r), {"instruction", "input", "output"})

    def test_reproducible_byte_identical(self):
        import generate_identity_data as gid
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "gen.jsonl"
            orig_out = gid.OUT
            gid.OUT = out
            try:
                gid.main()
            finally:
                gid.OUT = orig_out
            gen = out.read_bytes()
            committed = (ROOT / "data" / "monico_identity.jsonl").read_bytes()
            self.assertEqual(gen, committed, "regeneration must be byte-identical (seeded)")

    def test_categories_balanced(self):
        rows = [json.loads(l) for l in (ROOT / "data" / "monico_identity.jsonl")
                .read_text().splitlines() if l.strip()]
        instrs = [r["instruction"] for r in rows]
        # identity questions all present
        for q in ("Who are you?", "Are you Muse?", "Who made you?"):
            self.assertTrue(any(q in i for i in instrs), q)
        # coding rows present
        self.assertTrue(any("Python" in i for i in instrs))


class TestNotebookCommittedMatchesBuilder(unittest.TestCase):
    def test_committed_notebook_is_fresh_build(self):
        nbformat = __import__("nbformat")
        with tempfile.TemporaryDirectory() as td:
            out = str(Path(td) / "fresh.ipynb")
            rc = subprocess.run([sys.executable, str(ROOT / "build_notebook.py"),
                                 "--out", out], capture_output=True, text=True)
            self.assertEqual(rc.returncode, 0)
            fresh = nbformat.read(out, as_version=4)
            committed = nbformat.read(str(ROOT / "monico_train_colab.ipynb"), as_version=4)
            self.assertEqual(len(fresh.cells), len(committed.cells),
                             "committed notebook cell count differs from builder output — "
                             "re-run build_notebook.py and commit")
            for a, b in zip(fresh.cells, committed.cells):
                self.assertEqual(a.source, b.source)


if __name__ == "__main__":
    unittest.main()
