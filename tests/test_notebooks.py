"""Run notebook analysis cells in order without modifying research outputs."""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class NotebookWorkflowTests(unittest.TestCase):
    def test_comparison_run_all_preserves_correct_seed_statistics(self):
        notebook = json.loads((ROOT / "notebooks/model_comparison_final64.ipynb").read_text())
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            (work / "data/final_64").mkdir(parents=True)
            for source in (ROOT / "outputs/final_64").rglob("*.csv"):
                destination = work / source.relative_to(ROOT)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            namespace = {"display": lambda *args, **kwargs: None}
            previous = Path.cwd()
            try:
                os.chdir(work)
                with contextlib.redirect_stdout(io.StringIO()):
                    for index, cell in enumerate(notebook["cells"]):
                        if cell["cell_type"] == "code":
                            exec(
                                compile(
                                    "".join(cell["source"]), f"comparison cell {index}", "exec"
                                ),
                                namespace,
                            )
            finally:
                os.chdir(previous)
            result = pd.read_csv(
                work / "outputs/final_64/model_comparison/final_5seed_overall_comparison.csv"
            )
            expected = pd.read_csv(
                ROOT / "outputs/final_64/model_comparison/final_5seed_overall_comparison.csv"
            )
            pd.testing.assert_frame_equal(result, expected, rtol=1e-10, atol=1e-10)

    def test_training_benchmark_resolves_source_notebook(self):
        notebook = json.loads(
            (ROOT / "notebooks/training_cost_benchmark_final64.ipynb").read_text()
        )
        namespace = {"ROOT": ROOT}
        # The inspection cell reads the real GWN file but does not execute training.
        source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code" and "RELEVANT TRAINING CELLS" in "".join(cell["source"])
        )
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, "benchmark source path", "exec"), namespace)
        self.assertTrue(namespace["GWN_NOTEBOOK_PATH"].is_file())


if __name__ == "__main__":
    unittest.main()
