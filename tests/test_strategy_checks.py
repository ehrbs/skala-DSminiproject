"""Checks for irregular sampling, future-data exclusion and fixed policy splits."""
from pathlib import Path
import sys,unittest
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import strategy_checks as checks

class StrategyChecks(unittest.TestCase):
    def test_irregular_sampling_and_discharge_exclusion(self):
        result=checks.interval_features([0,9,9,10,10,11],[1,1,5,5,-2,-2])
        np.testing.assert_allclose([result[k] for k in checks.CURRENT],[1.4,np.sqrt(3.4),5,.1])
        self.assertIsNone(checks.interval_features([0,0,1],[-1,-1,-1]))

    def test_future_and_diagnostic_columns_are_rejected(self):
        frame=pd.DataFrame({'log_var_delta_q':[-4.], 'cycle_life':[1000]})
        self.assertEqual(list(checks.model_inputs(frame,['log_var_delta_q'])),['log_var_delta_q'])
        for name in ['cycle_life','knee_cycle','last_cycle','eol_not_observed','cell_id']:
            with self.assertRaises(ValueError):checks.model_inputs(frame,[name])

    def test_common_early_window_has_complete_coverage(self):
        cur=pd.read_csv(ROOT/'data/processed/early_current_features.csv')
        self.assertEqual(len(cur),139);self.assertEqual(cur.cell_id.nunique(),139)
        self.assertTrue((cur.early_cycles_used==4).all());self.assertTrue((cur.max_cycle_used==5).all())
        self.assertTrue((cur.cycle_window=='2-5').all());self.assertTrue(cur[checks.CURRENT].notna().all().all())
        self.assertTrue(cur.charge_fraction_gt4a.between(0,1).all())
        self.assertTrue((cur.charge_rms_a>=cur.charge_mean_a-1e-12).all())

    def test_policy_groups_do_not_overlap(self):
        split=pd.read_csv(ROOT/'data/processed/batch1_split_plan.csv')
        train=split[split.partition=='train_cv'];valid=split[split.partition=='holdout']
        self.assertEqual((len(train),len(valid)),(29,7))
        self.assertFalse(set(train.policy_group)&set(valid.policy_group))
        self.assertEqual(set(train.cv_fold),set(range(1,6)))
        for fold in range(1,6):
            self.assertFalse(set(train.loc[train.cv_fold==fold,'policy_group'])&set(train.loc[train.cv_fold!=fold,'policy_group']))

if __name__=='__main__':unittest.main()
