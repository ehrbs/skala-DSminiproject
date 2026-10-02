"""Independent checks of reported metrics, frozen split, and external predictions."""
from pathlib import Path
import json
import sys
import unittest

import numpy as np
import pandas as pd
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import day2_modeling as model
from strategy_checks import model_inputs


class Day2ModelingTests(unittest.TestCase):
    def test_fixed_partitions_and_early_feature_sets(self):
        train, valid, external, _ = model.load_data(ROOT)
        self.assertEqual((len(train), len(valid), len(external["Batch 2"]), len(external["Batch 3"])), (29, 7, 39, 44))
        self.assertFalse(set(train.policy_group) & set(valid.policy_group))
        for fold in range(1, 6):
            v = train.cv_fold == fold
            self.assertFalse(set(train.loc[v, "policy_group"]) & set(train.loc[~v, "policy_group"]))
        for features in model.FEATURE_SETS.values():
            self.assertEqual(list(model_inputs(train, features)), features)
        self.assertNotIn("knee_cycle", sum(model.FEATURE_SETS.values(), []))
        self.assertNotIn("cycle_life", sum(model.FEATURE_SETS.values(), []))

    def test_reported_scores_recompute_from_cell_predictions(self):
        prediction = pd.read_csv(ROOT / "results/day2_predictions.csv")
        summary = pd.read_csv(ROOT / "results/day2_performance.csv").set_index("index")
        for phase, index in [("Valid", "Valid (Batch 1 Hold-out)"), ("Batch 2", "Test (Batch 2)"), ("Batch 3", "Test (Batch 3)")]:
            cohort = prediction[prediction.phase == phase]
            recomputed = model.scores(cohort.actual, cohort.predicted)
            for name, value in recomputed.items():
                self.assertAlmostEqual(summary.loc[index, name], value, places=8)
        folds = pd.read_csv(ROOT / "results/day2_cv_folds.csv")
        chosen = json.loads((ROOT / "results/day2_selected_model.json").read_text())["candidate_id"]
        self.assertAlmostEqual(summary.loc["Train (Batch 1 CV)", "mape_pct"], folds.loc[folds.candidate == chosen, "mape_pct"].mean(), places=8)
        self.assertAlmostEqual(summary.loc["Gap (Train-Valid)", "mape_pct"], summary.loc["Valid (Batch 1 Hold-out)", "mape_pct"] - summary.loc["Train (Batch 1 CV)", "mape_pct"], places=8)
        self.assertAlmostEqual(summary.loc["Gap (Target-Test)", "mape_pct"], summary.loc["Test (Batch 2)", "mape_pct"] - 9.1, places=8)

    def test_selected_model_and_holdout_pred_match_refit(self):
        train, valid, external, _ = model.load_data(ROOT)
        chosen = json.loads((ROOT / "results/day2_selected_model.json").read_text())
        candidates = pd.read_csv(ROOT / "results/day2_candidates.csv")
        self.assertEqual(chosen["candidate_id"], candidates.iloc[0].id)
        spec = next(s for s in model.specifications() if s["id"] == chosen["candidate_id"])
        saved = pd.read_csv(ROOT / "results/day2_predictions.csv")
        fitted = clone(spec["estimator"]).fit(model_inputs(train, spec["features"]), train.cycle_life)
        expected = np.maximum(fitted.predict(model_inputs(valid, spec["features"])), 1.0)
        actual = valid[["cell_id"]].merge(saved.loc[saved.phase == "Valid", ["cell_id", "predicted"]], on="cell_id", validate="one_to_one").predicted.to_numpy()
        np.testing.assert_allclose(expected, actual, rtol=1e-12)
        full = pd.concat([train, valid], ignore_index=True)
        fitted = clone(spec["estimator"]).fit(model_inputs(full, spec["features"]), full.cycle_life)
        batch2 = external["Batch 2"]
        expected = np.maximum(fitted.predict(model_inputs(batch2, spec["features"])), 1.0)
        actual = batch2[["cell_id"]].merge(saved.loc[saved.phase == "Batch 2", ["cell_id", "predicted"]], on="cell_id", validate="one_to_one").predicted.to_numpy()
        np.testing.assert_allclose(expected, actual, rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
