"""Load processed cell data and enforce the frozen policy-group split."""
from pathlib import Path
import json
import pandas as pd
from features import FEATURE_SETS

def group_key(frame):
    return frame[["c1", "soc_switch", "c2"]].apply(
        lambda r: "|".join(f"{v:g}" for v in r), axis=1
    )


def load_data(root):
    root = Path(root)
    cells = pd.read_csv(root / "data/processed/cell_features.csv")
    current = pd.read_csv(root / "data/processed/early_current_features.csv")
    split = pd.read_csv(root / "data/processed/batch1_split_plan.csv")
    plan = json.loads((root / "data/processed/modeling_plan.json").read_text())
    assert len(cells) == 139 and cells.cell_id.is_unique
    assert len(current) == 139 and current.cell_id.is_unique
    data = cells.merge(current.drop(columns="batch"), on="cell_id", validate="one_to_one")
    assert set(sum(FEATURE_SETS.values(), [])) <= set(plan["input_whitelist"])
    assert (data.early_cycles_used == 4).all() and (data.max_cycle_used == 5).all()
    assert len(split) == plan["batch1_main_n"] == 36
    train = data.merge(split.loc[split.partition == "train_cv", ["cell_id", "cv_fold", "policy_group"]], on="cell_id", validate="one_to_one").sort_values("cell_id").reset_index(drop=True)
    holdout = data.merge(split.loc[split.partition == "holdout", ["cell_id", "policy_group"]], on="cell_id", validate="one_to_one").sort_values("cell_id").reset_index(drop=True)
    eligible = data.loc[data.cycle_life.notna() & (data.cycle_life > 0) & ~data.eol_not_observed]
    external = {}
    for batch in ("Batch 2", "Batch 3"):
        frame = eligible.loc[eligible.batch == batch].sort_values("cell_id").copy().reset_index(drop=True)
        frame["policy_group"] = group_key(frame)
        external[batch] = frame
    assert (len(train), len(holdout), len(external["Batch 2"]), len(external["Batch 3"])) == (29, 7, 39, 44)
    assert set(train.policy_group).isdisjoint(holdout.policy_group)
    assert set(train.cv_fold.astype(int)) == set(range(1, 6))
    for fold in range(1, 6):
        v = train.cv_fold == fold
        assert set(train.loc[v, "policy_group"]).isdisjoint(train.loc[~v, "policy_group"])
    assert train.cycle_life.min() == plan["train_target_min"]
    assert train.cycle_life.max() == plan["train_target_max"]
    return train, holdout, external, plan
