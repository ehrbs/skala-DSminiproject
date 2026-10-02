"""Check validation boundaries and saved-model predictions independently."""
from pathlib import Path
import json
import sys
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train import specifications, load_data
from evaluation import nested_group_cv, mape
from predict import predict_cells


class SupplementaryEvaluationTests(unittest.TestCase):
    def test_outer_labels_do_not_change_inner_selection(self):
        train, _, _, _ = load_data(ROOT)
        specs = [s for s in specifications() if s["feature_set"] == "G0_delta" and s["model"] == "Ridge" and s["target_scale"] == "raw"]
        original, _, _, _ = nested_group_cv(train, specs)
        changed = train.copy()
        changed.loc[changed.cv_fold.eq(1), "cycle_life"] *= 3
        perturbed, _, _, _ = nested_group_cv(changed, specs)
        a = original.set_index("outer_fold").loc[1]
        b = perturbed.set_index("outer_fold").loc[1]
        self.assertEqual(a.chosen_candidate, b.chosen_candidate)
        self.assertAlmostEqual(a.inner_selection_mape_pct, b.inner_selection_mape_pct, places=12)
        self.assertNotAlmostEqual(a.outer_mape_pct, b.outer_mape_pct)

    def test_saved_nested_audit_and_scores(self):
        audit = pd.read_csv(ROOT / "results/day2_nested_split_audit.csv")
        train, _, _, _ = load_data(ROOT)
        policy = train.set_index("cell_id").policy_group.to_dict()
        self.assertEqual(len(audit), 20)
        for row in audit.itertuples():
            fit, valid, outer = (set(x.split(";")) for x in (row.fit_cell_ids,row.valid_cell_ids,row.outer_valid_cell_ids))
            self.assertFalse(fit & valid)
            self.assertFalse((fit | valid) & outer)
            self.assertEqual(fit | valid | outer,set(train.cell_id))
            self.assertFalse({policy[x] for x in fit} & {policy[x] for x in valid})
            self.assertFalse({policy[x] for x in fit | valid} & {policy[x] for x in outer})
        pred = pd.read_csv(ROOT / "results/day2_nested_predictions.csv")
        folds = pd.read_csv(ROOT / "results/day2_nested_folds.csv")
        self.assertEqual(set(pred.cell_id),set(train.cell_id))
        self.assertTrue(pred.cell_id.is_unique)
        for row in folds.itertuples():
            p = pred.loc[pred.outer_fold.eq(row.outer_fold)]
            self.assertAlmostEqual(mape(p.actual,p.predicted),row.outer_mape_pct,places=10)
        summary = json.loads((ROOT / "results/day2_validation_summary.json").read_text())
        self.assertAlmostEqual(folds.outer_mape_pct.mean(),summary["nested_mape_mean_pct"],places=10)

    def test_saved_model_requires_only_initial_features(self):
        _, _, external, _ = load_data(ROOT)
        manifest = json.loads((ROOT / "results/day2_model_manifest.json").read_text())
        frame = external["Batch 2"][["cell_id",*manifest["features"]]].copy()
        p = predict_cells(ROOT,frame)
        saved = pd.read_csv(ROOT / "results/day2_predictions.csv")
        expected = saved.loc[saved.phase.eq("Batch 2")].set_index("cell_id").loc[p.cell_id,"predicted"]
        np.testing.assert_allclose(p.predicted_cycle_life,expected,rtol=1e-12)
        with_target = frame.assign(cycle_life=1.0,knee_cycle=100000.0)
        np.testing.assert_array_equal(p.predicted_cycle_life,predict_cells(ROOT,with_target).predicted_cycle_life)
        invalid = frame.copy();invalid.loc[invalid.index[0],manifest["features"][0]] = np.inf
        with self.assertRaises(ValueError):
            predict_cells(ROOT,invalid)
        numeric_equation = manifest["raw_input_intercept_cycles"] + sum(frame[k]*v for k,v in manifest["raw_input_coefficients"].items())
        np.testing.assert_allclose(p.predicted_cycle_life,np.maximum(numeric_equation,1),rtol=1e-12)

    def test_risk_and_guide_summary_match_predictions(self):
        pred = pd.read_csv(ROOT / "results/day2_predictions.csv")
        risk = pd.read_csv(ROOT / "results/day2_short_life_risk.csv")
        for row in risk.itertuples():
            p = pred.loc[pred.phase.eq(row.phase)]
            self.assertEqual(row.short_missed_n,int(((p.actual<500) & (p.predicted>=500)).sum()))
        summary = pd.read_csv(ROOT / "results/day2_performance.csv")
        guide = pd.read_csv(ROOT / "results/model_performance.csv")
        self.assertEqual(list(guide["구분"]),list(summary["index"]))
        np.testing.assert_allclose(guide["MAPE (%)"],summary.mape_pct.round(2),rtol=0,atol=1e-12)


if __name__ == "__main__":
    unittest.main()
