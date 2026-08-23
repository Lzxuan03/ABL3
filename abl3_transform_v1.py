from __future__ import annotations

import numpy as np

from bandwise_drift_final_validation_v1.calibration.bandwise_drift_control_v1 import (
    BandwiseWelford,
    BandwiseWindow,
    band_drift,
    band_lambdas,
)
from causal_reliability_calibration_v1.online_statistics_v1 import normalize_with_stats


RETENTION_POLICY = "SHARED_MEAN_OF_CANONICAL_BAND_RETENTIONS"


def transform_a8_shared_retention(
    raw: np.ndarray,
    source_mean: np.ndarray,
    source_var: np.ndarray,
    *,
    warmup: int = 64,
    recent_window: int = 64,
    lambda_min: float = 0.98,
    drift_theta: float = 1.0,
    drift_temperature: float = 0.25,
    eta: float = 0.5,
    return_trace: bool = False,
):
    """Canonical A8 with per-band retention application replaced by its mean."""
    transformed = []
    state = BandwiseWelford()
    canonical_controller_state = BandwiseWelford()
    recent = BandwiseWindow(recent_window)
    traces = []
    state_updates = 0
    for window_index, xi in enumerate(np.asarray(raw, dtype=np.float64)):
        cumulative_mean, cumulative_var, counts_before = state.stats(source_mean, source_var)
        controller_mean, controller_var, _controller_counts = canonical_controller_state.stats(source_mean, source_var)
        recent_mean, recent_var, recent_count = recent.stats(source_mean, source_var)
        drift, _mean_drift, _var_drift = band_drift(
            controller_mean, controller_var, recent_mean, recent_var,
            eta=eta, use_mean=False, use_var=True,
        )
        canonical_retention = np.ones(5, dtype=np.float64)
        if recent_count >= 2 and window_index >= warmup:
            canonical_retention = band_lambdas(
                drift, drift_theta, drift_temperature, lambda_min,
            )
        shared_value = float(np.mean(canonical_retention, dtype=np.float64))
        applied_retention = np.full(5, shared_value, dtype=np.float64)
        output = normalize_with_stats(xi[None, :], cumulative_mean, cumulative_var)[0]
        state.update(xi, lambdas=applied_retention)
        canonical_controller_state.update(xi, lambdas=canonical_retention)
        recent.update(xi)
        state_updates += 1
        if return_trace:
            mean_after, var_after, counts_after = state.stats(source_mean, source_var)
            traces.append({
                "window": window_index,
                **{f"r_band_{band + 1}": float(canonical_retention[band]) for band in range(5)},
                "r_shared": shared_value,
                "mean_original": float(np.mean(canonical_retention, dtype=np.float64)),
                "abs_error": abs(shared_value - float(np.mean(canonical_retention, dtype=np.float64))),
                "band_retention_std": float(np.std(canonical_retention)),
                "applied_retention_std": float(np.std(applied_retention)),
                "state_count_before_mean": float(np.mean(counts_before)),
                "state_count_after_mean": float(np.mean(counts_after)),
                "state_mean": float(np.mean(mean_after)),
                "state_variance": float(np.mean(var_after)),
                "output_finite": bool(np.isfinite(output).all()),
                "predict_then_update": True,
            })
        transformed.append(output)
    value = np.asarray(transformed, dtype=np.float32)
    metadata = {
        "calibration_method": "A8_SHARED_RETENTION",
        "transform_a8_called": True,
        "retention_policy": RETENTION_POLICY,
        "state_updates_occur": state_updates == len(raw) and state_updates > 0,
        "state_update_count": state_updates,
        "predict_then_update": True,
        "recent_variance_comparator_computed": True,
        "warmup": int(warmup),
        "recent_window": int(recent_window),
        "output_finite": bool(np.isfinite(value).all()),
    }
    return value, metadata, traces
