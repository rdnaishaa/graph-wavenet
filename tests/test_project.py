"""Regression checks for the frozen Final64 artifacts and corrected training loss."""

import ast
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from streamlit_app.data_loader import get_target_values, load_test_data
from streamlit_app.graph_explorer import load_adaptive_graph
from streamlit_app.model_loader import load_graphwavenet_model

ROOT = Path(__file__).resolve().parents[1]


class Final64Tests(unittest.TestCase):
    def test_saved_models_accept_final64_input(self):
        sample = load_test_data()["X"][:1].astype(np.float32)
        for mode in ("fixed", "adaptive", "hybrid"):
            with self.subTest(mode=mode):
                prediction = load_graphwavenet_model(mode)(sample, training=False).numpy()
                self.assertEqual(prediction.shape, (1, 3, 64))
                self.assertTrue(np.isfinite(prediction).all())

    def test_target_display_respects_observation_mask(self):
        data = load_test_data()
        for sample in (0, len(data["X"]) - 1):
            result = get_target_values(sample, 0)
            valid = data["mask"][sample, :, 0].astype(bool)
            # Display values must never treat imputed targets as observed ground truth.
            self.assertEqual(int(result["actual_speed"].notna().sum()), int(valid.sum()))

    def test_graph_explorer_uses_current_gwn_artifact(self):
        matrix, source = load_adaptive_graph()
        self.assertIn("graphwavenet_audited", source)
        expected = np.load(
            ROOT / "outputs/final_64/graphwavenet_audited/"
            "graphwavenet_adaptive_seed42_adjacency.npy"
        )
        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_corrected_loss_ignores_masked_targets_and_trains(self):
        notebook = json.loads((ROOT / "notebooks/gclstm_final64_baseline.ipynb").read_text())
        definitions = []
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse("".join(cell["source"]))
            definitions.extend(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "masked_mse"
            )
        self.assertEqual(len(definitions), 1)
        namespace = {"tf": tf}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "loss", "exec"), namespace)
        loss = namespace["masked_mse"]
        values = tf.constant([[[2.0, 100.0]]])
        mask = tf.constant([[[1.0, 0.0]]])
        packed = tf.stack([values, mask], axis=-1)
        self.assertAlmostEqual(float(loss(packed, tf.zeros_like(values))), 4.0)
        empty = tf.stack([values, tf.zeros_like(mask)], axis=-1)
        self.assertEqual(float(loss(empty, tf.zeros_like(values))), 0.0)
        # Exercise Keras packed-target handling, not just the standalone formula.
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input((2,)),
                tf.keras.layers.Dense(2),
                tf.keras.layers.Reshape((1, 2)),
            ]
        )
        model.compile(optimizer="adam", loss=loss)
        self.assertTrue(np.isfinite(model.train_on_batch(np.ones((1, 2)), packed)))

    def test_overall_sd_is_computed_across_seed_scores(self):
        path = ROOT / "outputs/final_64/model_comparison"
        gwn = pd.read_csv(
            ROOT / "outputs/final_64/graphwavenet_audited/graphwavenet_multiseed_all_results.csv"
        ).rename(columns={"MAE": "mae", "RMSE": "rmse"})
        gwn["model_clean"] = gwn["mode"].map(
            {"fixed": "Fixed GWN", "adaptive": "Adaptive GWN", "hybrid": "Hybrid GWN"}
        )
        baseline = pd.read_csv(
            ROOT / "outputs/final_64/gclstm_baseline/gclstm_fair_multiseed_results.csv"
        )
        baseline["model_clean"] = "GC-LSTM Fair"
        raw = pd.concat([gwn, baseline], ignore_index=True)
        overall = pd.read_csv(path / "final_5seed_overall_comparison.csv").set_index("model_clean")
        scores = raw.groupby(["model_clean", "seed"])[["mae", "rmse"]].mean()
        expected = scores.groupby("model_clean").agg(["mean", "std"])
        for model in expected.index:
            for metric in ("mae", "rmse"):
                self.assertAlmostEqual(
                    overall.loc[model, f"avg_{metric}"], expected.loc[model, (metric, "mean")]
                )
                self.assertAlmostEqual(
                    overall.loc[model, f"avg_{metric}_std"], expected.loc[model, (metric, "std")]
                )


if __name__ == "__main__":
    unittest.main()
