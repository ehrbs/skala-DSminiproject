"""Early-cycle feature groups fixed before DAY 2 model selection."""
from strategy_checks import model_inputs

FEATURE_SETS = {
    "G0_delta": ["log_var_delta_q"],
    "G1_trend": ["log_var_delta_q", "qd_slope_10_100"],
    "G2_charge": ["log_var_delta_q", "qd_slope_10_100", "mean_chargetime", "c1"],
    "G3_thermal": ["log_var_delta_q", "qd_slope_10_100", "mean_chargetime", "c1", "mean_tavg"],
    "G4_current": ["log_var_delta_q", "charge_rms_a"],
}

def feature_matrix(frame, names):
    """Return only allowed early-cycle inputs, excluding labels and future data."""
    return model_inputs(frame, names)
