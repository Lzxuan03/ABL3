from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from causal_hierarchical_fusion_final_v1.inference_utils_v1 import transform_a8
from paper_faithful_dgcnn_dual_protocol_v1.data.paper_faithful_eeg_loader_v1 import load_subject_trials
from strict_cross_dgcnn_sourceval_seed2022_v1.data.build_sourceval_cross_v1 import validation_subject_for

from replacement_a8_strict_cross_multiseed_confirmation_v1.common_v1 import SUBJECTS, stable_hash


def split_definition(dataset: str, target: int, seed: int) -> dict[str, Any]:
    subjects = SUBJECTS[dataset]
    validation_subject, rng_seed = validation_subject_for(dataset, seed, target, subjects)
    training_subjects = [s for s in subjects if s not in {target, validation_subject}]
    value = {
        "dataset": dataset,
        "seed": int(seed),
        "outer_target": int(target),
        "training_subjects": training_subjects,
        "validation_subject": int(validation_subject),
        "validation_rng_seed": int(rng_seed),
        "target_subject": int(target),
        "selection_rule": "DETERMINISTIC_SEEDED_SOURCE_HOLDOUT_V1",
    }
    value["split_hash"] = stable_hash(value)
    return value


def _load_subject(dataset: str, subject: int) -> list[dict[str, Any]]:
    trials, _ = load_subject_trials(dataset, subject)
    rows = []
    for trial in trials:
        row = dict(trial)
        row["subject"] = int(subject)
        row["x"] = np.asarray(row["x"], dtype=np.float32)
        rows.append(row)
    return rows


def _transform_trials(trials: list[dict[str, Any]], scaler: StandardScaler) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        row = dict(trial)
        row["x"] = scaler.transform(trial["x"]).astype(np.float32).reshape(-1, 62, 5)
        rows.append(row)
    return rows


def flatten_windows(trials: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    output: dict[str, list[np.ndarray]] = {
        "x": [], "y": [], "subject": [], "session": [], "trial": [], "window_index": []
    }
    for trial in trials:
        count = len(trial["x"])
        output["x"].append(trial["x"].reshape(count, 310))
        output["y"].append(np.full(count, int(trial["label"]), dtype=np.int64))
        output["subject"].append(np.full(count, int(trial["subject"]), dtype=np.int64))
        output["session"].append(np.full(count, int(trial["session"]), dtype=np.int64))
        output["trial"].append(np.full(count, int(trial["trial"]), dtype=np.int64))
        output["window_index"].append(np.arange(count, dtype=np.int64))
    return {key: np.concatenate(value) for key, value in output.items()}


def _add_a8(windows: dict[str, np.ndarray], source_mean: np.ndarray, source_var: np.ndarray):
    output = dict(windows)
    transformed = np.empty_like(windows["x"], dtype=np.float32)
    units = np.stack([windows["subject"], windows["session"]], axis=1)
    for unit in np.unique(units, axis=0):
        index = np.where((units == unit).all(axis=1))[0]
        transformed[index], _ = transform_a8(windows["x"][index], source_mean, source_var)
    output["x_a8"] = transformed
    return output


def build_source(dataset: str, target: int, seed: int):
    definition = split_definition(dataset, target, seed)
    train_raw = [trial for subject in definition["training_subjects"] for trial in _load_subject(dataset, subject)]
    validation_raw = _load_subject(dataset, definition["validation_subject"])
    scaler_rows = np.concatenate([trial["x"] for trial in train_raw], axis=0)
    scaler = StandardScaler().fit(scaler_rows)
    normalization_hash = stable_hash(np.concatenate([scaler.mean_, scaler.scale_]).astype(np.float64))
    normalization = {
        "normalization_hash": normalization_hash,
        "fit_subjects": definition["training_subjects"],
        "fit_rows": int(len(scaler_rows)),
        "fit_axis": "310 flattened channel-band coordinates",
        "validation_used_in_fit": False,
        "target_used_in_fit": False,
        "mean_finite": bool(np.isfinite(scaler.mean_).all()),
        "scale_finite": bool(np.isfinite(scaler.scale_).all()),
    }
    train = flatten_windows(_transform_trials(train_raw, scaler))
    validation = flatten_windows(_transform_trials(validation_raw, scaler))
    source_mean = train["x"].mean(axis=0).astype(np.float64)
    source_var = np.maximum(train["x"].var(axis=0), 1e-5).astype(np.float64)
    return (
        _add_a8(train, source_mean, source_var),
        _add_a8(validation, source_mean, source_var),
        scaler,
        definition,
        normalization,
        source_mean,
        source_var,
    )


def load_target_once(dataset: str, target: int, scaler, source_mean, source_var):
    target_trials = _transform_trials(_load_subject(dataset, target), scaler)
    return _add_a8(flatten_windows(target_trials), source_mean, source_var)
