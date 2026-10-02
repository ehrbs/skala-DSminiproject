"""Apply the saved early-cycle model to a cell feature CSV."""
from pathlib import Path
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from features import feature_matrix


def predict_cells(root, frame):
    root = Path(root)
    manifest = json.loads((root / "results/day2_model_manifest.json").read_text())
    model = joblib.load(root / "results/day2_model.joblib")
    x = feature_matrix(frame, manifest["features"])
    if not all(pd.api.types.is_numeric_dtype(x[c]) for c in x):
        raise ValueError("Model inputs must be numeric early-cycle features")
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values are invalid")
    output = frame[["cell_id"]].copy() if "cell_id" in frame else pd.DataFrame(index=frame.index)
    output["predicted_cycle_life"] = np.maximum(model.predict(x), manifest["prediction_min_cycles"])
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    result = predict_cells(args.root, pd.read_csv(args.input))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(destination,index=False)
    print(f"Saved predictions for {len(result)} cells to {destination}")
