from __future__ import annotations

import re
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from data.label_provider import get_seed_iv_labels, get_seed_labels, get_seed_v_labels
from data.split_protocol import get_within_subject_split


class DataLoadError(RuntimeError):
    """Raised when dataset feature files cannot be resolved or parsed."""


_PROJECT_ROOT = Path(__file__).resolve().parent
#_DEFAULT_SEED_V_ROOT = Path(r"F:\EEG数据集\情感\SEED-V")
_DEFAULT_SEED_V_ROOT = Path(r"/home/EEG/SEED-V")


def _data_debug_enabled() -> bool:
    return os.environ.get("CMCRD_DATA_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_log(msg: str) -> None:
    if _data_debug_enabled():
        print(msg)

def _default_data_root() -> Path:
    return (_PROJECT_ROOT / "datasets").resolve()


def _default_seed_v_root() -> Path:
    """Default raw SEED-V location. Override with SEED_V_ROOT if needed."""
    env_root = os.environ.get("SEED_V_ROOT")
    return Path(env_root).expanduser().resolve() if env_root else _DEFAULT_SEED_V_ROOT


def _normalize_subject(subject: int) -> str:
    return str(int(subject))


def _extract_subject_session(path: Path) -> tuple[int | None, int | None]:
    name = path.stem.lower()
    subject = None
    session = None

    sub_match = re.search(r"(?:subject|sub|s)[_\- ]*0*(\d{1,2})", name)
    ses_match = re.search(r"(?:session|sess|ses)[_\- ]*0*(\d)", name)
    if sub_match:
        subject = int(sub_match.group(1))
    if ses_match:
        session = int(ses_match.group(1))

    if subject is None:
        numbers = [int(x) for x in re.findall(r"\d+", name)]
        if numbers:
            subject = numbers[0]
        if len(numbers) >= 2 and session is None and numbers[1] in (1, 2, 3):
            session = numbers[1]
    return subject, session


def _seed_v_official_feature_paths(seed_v_root: Path, subject: int) -> tuple[Path, Path] | None:
    eeg_path = seed_v_root / "EEG_DE_features" / f"{int(subject)}_123.npz"
    eye_path = seed_v_root / "Eye_movement_features" / f"{int(subject)}_123.npz"
    if eeg_path.exists() and eye_path.exists():
        return eeg_path, eye_path
    return None


def _load_pickled_npz_dict(path: Path, key: str) -> dict[Any, Any]:
    _debug_log(f"[DEBUG] _load_npz path = {path}")
    obj = np.load(path, allow_pickle=True)
    if key not in obj.files:
        raise DataLoadError(f"Expected key='{key}' in official SEED-V file: {path}")
    value = obj[key]
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, bytearray, np.bytes_)):
        # NumPy 2.4 may raise a deprecation warning during unpickling of legacy dtypes.
        # This warning is noisy but does not affect loaded values here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            warnings.filterwarnings(
                "ignore",
                message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
            )
            value = pickle.loads(value)
    if not isinstance(value, dict):
        raise DataLoadError(f"Expected pickled dict at key='{key}' in {path}, got {type(value)}")
    return value


def _load_seed_v_official_subject_session(
    seed_v_root: Path, subject: int, session: int, official_trial_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = _seed_v_official_feature_paths(seed_v_root, subject)
    if paths is None:
        raise DataLoadError(
            f"Official SEED-V EEG/Eye feature pair not found for subject={subject} under {seed_v_root}."
        )

    eeg_path, eye_path = paths
    eeg_data = _load_pickled_npz_dict(eeg_path, "data")
    eye_data = _load_pickled_npz_dict(eye_path, "data")
    eeg_label = _load_pickled_npz_dict(eeg_path, "label")

    x_list = []
    y_list = []
    t_list = []
    start = (int(session) - 1) * 15
    for global_trial in range(start, start + 15):
        if global_trial not in eeg_data or global_trial not in eye_data:
            raise DataLoadError(
                f"Missing trial key={global_trial} for subject={subject}, session={session}."
            )

        eeg = _to_2d(np.asarray(eeg_data[global_trial]), eeg_path, f"data[{global_trial}]")
        eye = _to_2d(np.asarray(eye_data[global_trial]), eye_path, f"data[{global_trial}]")
        if eeg.shape[0] != eye.shape[0]:
            raise DataLoadError(
                f"EEG/Eye row mismatch for subject={subject}, session={session}, "
                f"trial={global_trial}: eeg={eeg.shape}, eye={eye.shape}"
            )

        trial_id = global_trial - start
        x_list.append(np.concatenate([eeg, eye], axis=1).astype(np.float32))
        if global_trial in eeg_label:
            y = np.asarray(eeg_label[global_trial]).reshape(-1).astype(np.int64)
            if len(y) != eeg.shape[0]:
                y = np.full(eeg.shape[0], official_trial_labels[trial_id], dtype=np.int64)
        else:
            y = np.full(eeg.shape[0], official_trial_labels[trial_id], dtype=np.int64)
        y_list.append(y.reshape(-1, 1))
        t_list.append(np.full(eeg.shape[0], trial_id, dtype=np.int64))

    return np.concatenate(x_list, axis=0), np.concatenate(y_list, axis=0), np.concatenate(t_list, axis=0)

'''
def _find_feature_files(seed_v_root: Path) -> list[Path]:
    files = []
    for ext in ("*.npz", "*.npy", "*.mat"):
        files.extend(seed_v_root.rglob(ext))
    files = [p for p in files if "emotion_label_and_stimuli_order" not in p.name.lower()]
    return sorted(files)
'''
def _find_feature_files(seed_v_root: Path) -> list[Path]:
    files = []

    # 官方 SEED-V 特征
    eeg_dir = seed_v_root / "EEG_DE_features"
    eye_dir = seed_v_root / "Eye_movement_features"

    if eeg_dir.exists():
        files.extend(sorted(eeg_dir.glob("*.npz")))
    if eye_dir.exists():
        files.extend(sorted(eye_dir.glob("*.npz")))

    return sorted(files)

def _parse_key_candidates(mapping: dict[str, Any], keys: tuple[str, ...]) -> Any | None:
    lower_map = {k.lower(): k for k in mapping}
    for key in keys:
        if key in lower_map:
            return mapping[lower_map[key]]
    return None


def _infer_trial_ids(num_rows: int, num_trials: int = 15) -> np.ndarray:
    if num_rows < num_trials:
        raise DataLoadError(
            f"Cannot infer trial ids: num_rows={num_rows} < num_trials={num_trials}."
        )
    base = np.arange(num_trials)
    repeat = int(np.ceil(num_rows / num_trials))
    return np.tile(base, repeat)[:num_rows]


def _to_2d(arr: np.ndarray, source: Path, name: str) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise DataLoadError(
            f"Expected 2D array for '{name}' in {source}, got shape={arr.shape}."
        )
    return arr


def _load_npz(path: Path) -> dict[str, Any]:
    obj = np.load(path, allow_pickle=True)
    if hasattr(obj, "files"):
        return {k: obj[k] for k in obj.files}
    return {"data": obj}


def _load_npy(path: Path) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.shape == ():
        val = arr.item()
        if isinstance(val, dict):
            return val
    if isinstance(arr, dict):
        return arr
    return {"data": arr}


def _load_mat(path: Path) -> dict[str, Any]:
    raw = loadmat(path)
    return {k: v for k, v in raw.items() if not k.startswith("__")}

'''
def _read_file_to_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        print(f"[DEBUG] _read_file_to_mapping path = {path}")
        return _load_npz(path)
    if suffix == ".npy":
        print(f"[DEBUG] _read_file_to_mapping path = {path}")
        return _load_npz(path)
    if suffix == ".mat":
        print(f"[DEBUG] _read_file_to_mapping path = {path}")
        return _load_mat(path)
    raise DataLoadError(f"Unsupported file type: {path}")
'''
def _read_file_to_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return _load_npz(path)
    if suffix == ".npy":
        return _load_npy(path)
    if suffix == ".mat":
        return _load_mat(path)
    raise DataLoadError(f"Unsupported file type: {path}")

def _extract_xy_trial(
    mapping: dict[str, Any], path: Path, official_trial_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = _parse_key_candidates(mapping, ("data", "x", "feature", "features", "de_feature"))
    label = _parse_key_candidates(mapping, ("label", "y", "labels", "target"))
    trial = _parse_key_candidates(mapping, ("trial", "trial_id", "trial_index", "video_id", "clip_id"))

    if data is not None:
        data = np.asarray(data)
        if data.ndim == 3 and data.shape[0] == 15:
            x = np.concatenate([_to_2d(data[i], path, f"data[{i}]") for i in range(15)], axis=0)
            trial_idx = np.concatenate(
                [np.full(_to_2d(data[i], path, f"data[{i}]").shape[0], i, dtype=np.int64) for i in range(15)]
            )
            y = official_trial_labels[trial_idx]
            return x.astype(np.float32), y.reshape(-1, 1).astype(np.int64), trial_idx

        x = _to_2d(data, path, "data").astype(np.float32)
        if trial is not None:
            trial_idx = np.asarray(trial).reshape(-1).astype(int)
            if len(trial_idx) != len(x):
                raise DataLoadError(
                    f"trial_index length mismatch in {path}: "
                    f"len(trial_index)={len(trial_idx)}, len(data)={len(x)}"
                )
            if trial_idx.min() == 1:
                trial_idx = trial_idx - 1
        else:
            trial_idx = _infer_trial_ids(len(x), num_trials=15)

        if label is not None:
            y = np.asarray(label).reshape(-1).astype(int)
            if len(y) != len(x):
                raise DataLoadError(
                    f"label length mismatch in {path}: len(label)={len(y)}, len(data)={len(x)}"
                )
            unique = np.unique(y)
            if unique.min() < 0 or unique.max() > 4:
                # Normalize labels that may not be 0-based.
                _, y = np.unique(y, return_inverse=True)
            y = y.astype(np.int64)
        else:
            y = official_trial_labels[trial_idx].astype(np.int64)
        return x, y.reshape(-1, 1), trial_idx.astype(np.int64)

    trial_arrays = []
    for key, value in mapping.items():
        m = re.search(r"(\d{1,2})", key.lower())
        if not m:
            continue
        tid = int(m.group(1))
        if 1 <= tid <= 15:
            arr = np.asarray(value)
            if arr.size == 0:
                continue
            trial_arrays.append((tid - 1, _to_2d(arr, path, key)))

    if not trial_arrays:
        keys = list(mapping.keys())
        raise DataLoadError(
            f"Could not parse feature array from {path}. "
            f"Available keys: {keys[:20]}"
        )

    trial_arrays.sort(key=lambda item: item[0])
    x_list = []
    trial_idx_list = []
    for tid, arr in trial_arrays:
        x_list.append(arr.astype(np.float32))
        trial_idx_list.append(np.full(arr.shape[0], tid, dtype=np.int64))
    x = np.concatenate(x_list, axis=0)
    trial_idx = np.concatenate(trial_idx_list, axis=0)
    y = official_trial_labels[trial_idx]
    return x, y.reshape(-1, 1).astype(np.int64), trial_idx


def _collect_subject_session_data(
    seed_v_root: Path, subject: int, session: int, official_trial_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if session in (1, 2, 3) and _seed_v_official_feature_paths(seed_v_root, subject) is not None:
        return _load_seed_v_official_subject_session(
            seed_v_root, subject, session, official_trial_labels
        )

    files = _find_feature_files(seed_v_root)
    if not files:
        raise DataLoadError(
            f"No feature files (.npz/.npy/.mat) found under {seed_v_root}.\n"
            "Expected official feature files plus emotion_label_and_stimuli_order.xlsx."
        )

    subj = int(subject)
    ses = int(session)
    selected = []
    for path in files:
        p_sub, p_ses = _extract_subject_session(path)
        if p_sub is not None and p_sub != subj:
            continue
        if p_ses is not None and ses in (1, 2, 3) and p_ses != ses:
            continue
        selected.append(path)

    if not selected:
        raise DataLoadError(
            f"No feature file matched subject={subject}, session={session} under {seed_v_root}."
        )

    x_list = []
    y_list = []
    t_list = []
    for path in selected:
        _debug_log(f"[DEBUG] loading file: {path}")
        mapping = _read_file_to_mapping(path)
        x, y, t = _extract_xy_trial(mapping, path, official_trial_labels)
        x_list.append(x)
        y_list.append(y)
        t_list.append(t)

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    t_all = np.concatenate(t_list, axis=0)
    return x_all, y_all, t_all


def _split_by_trial(
    x: np.ndarray, y: np.ndarray, trial_idx: np.ndarray, dataset_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = get_within_subject_split(dataset_name, total_trials=15)
    tr_mask = np.isin(trial_idx, np.asarray(split.train_trials))
    te_mask = np.isin(trial_idx, np.asarray(split.test_trials))
    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        raise DataLoadError(
            "Trial split resulted in empty train/test set. "
            "Please verify your feature file trial indexing."
        )
    return x[tr_mask], y[tr_mask], x[te_mask], y[te_mask]


def _split_by_trial_with_ids(
    x: np.ndarray, y: np.ndarray, trial_idx: np.ndarray, dataset_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    split = get_within_subject_split(dataset_name, total_trials=15)
    tr_mask = np.isin(trial_idx, np.asarray(split.train_trials))
    te_mask = np.isin(trial_idx, np.asarray(split.test_trials))
    if tr_mask.sum() == 0 or te_mask.sum() == 0:
        raise DataLoadError(
            "Trial split resulted in empty train/test set. "
            "Please verify your feature file trial indexing."
        )
    return (
        x[tr_mask],
        y[tr_mask],
        x[te_mask],
        y[te_mask],
        trial_idx[tr_mask],
        trial_idx[te_mask],
    )


def load_SEED_V_within_session_data(
    subject: int,
    session: int = 4,
    data_root: str | Path | None = None,
    return_trial_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SEED-V features and labels for within-subject protocol.
    session=1/2/3 loads one session; session=4 concatenates sessions if available.
    """
    root = Path(data_root) if data_root else _default_seed_v_root()

    if session in (1, 2, 3):
        labels, _ = get_seed_v_labels(root, session_id=session)
        x, y, trial_idx = _collect_subject_session_data(root, subject, session, labels)
    elif session == 4:
        x_parts, y_parts, t_parts = [], [], []
        for ses in (1, 2, 3):
            try:
                labels, _ = get_seed_v_labels(root, session_id=ses)
                x_s, y_s, t_s = _collect_subject_session_data(root, subject, ses, labels)
            except DataLoadError:
                continue
            x_parts.append(x_s)
            y_parts.append(y_s)
            t_parts.append(t_s)
        if not x_parts:
            # fallback: files without explicit session marks
            labels, _ = get_seed_v_labels(root, session_id=1)
            x_s, y_s, t_s = _collect_subject_session_data(root, subject, 1, labels)
            x_parts, y_parts, t_parts = [x_s], [y_s], [t_s]
        x = np.concatenate(x_parts, axis=0)
        y = np.concatenate(y_parts, axis=0)
        trial_idx = np.concatenate(t_parts, axis=0)
    else:
        raise ValueError(f"Unsupported session={session}. Use 1/2/3/4.")

    if return_trial_ids:
        return _split_by_trial_with_ids(x, y, trial_idx, "seed_v")

    train_x, train_y, test_x, test_y = _split_by_trial(x, y, trial_idx, "seed_v")
    return train_x, train_y, test_x, test_y


def load_SEED_V_cross_subject_data(
    subject: int,
    session: int = 4,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Leave-one-subject-out on SEED-V.
    Returns train_data/train_label from all subjects except `subject`,
    and test_data/test_label from the held-out subject.
    """
    root = Path(data_root) if data_root else _default_seed_v_root()

    files = _find_feature_files(root)
    if not files:
        raise DataLoadError(f"No feature files found under {root}.")
    subject_ids = sorted(
        {
            sid
            for sid, _ in (_extract_subject_session(p) for p in files)
            if sid is not None and sid > 0
        }
    )
    if not subject_ids:
        raise DataLoadError(
            "Could not infer subject ids from filenames. "
            "Please include subject index in file names, e.g., subject01_session1.npz."
        )

    x_train_parts, y_train_parts = [], []
    x_test, y_test = None, None

    for sid in subject_ids:
        if session in (1, 2, 3):
            labels, _ = get_seed_v_labels(root, session_id=session)
            x, y, _ = _collect_subject_session_data(root, sid, session, labels)
        else:
            x_all, y_all = [], []
            for ses in (1, 2, 3):
                try:
                    labels, _ = get_seed_v_labels(root, session_id=ses)
                    xs, ys, _ = _collect_subject_session_data(root, sid, ses, labels)
                except DataLoadError:
                    continue
                x_all.append(xs)
                y_all.append(ys)
            if not x_all:
                labels, _ = get_seed_v_labels(root, session_id=1)
                x, y, _ = _collect_subject_session_data(root, sid, 1, labels)
            else:
                x = np.concatenate(x_all, axis=0)
                y = np.concatenate(y_all, axis=0)

        if sid == int(subject):
            x_test, y_test = x, y
        else:
            x_train_parts.append(x)
            y_train_parts.append(y)

    if x_test is None or y_test is None:
        raise DataLoadError(f"Held-out subject={subject} was not found in data under {root}.")

    if not x_train_parts:
        raise DataLoadError("Training set is empty after leave-one-subject-out split.")

    train_x = np.concatenate(x_train_parts, axis=0)
    train_y = np.concatenate(y_train_parts, axis=0)
    return train_x, train_y, x_test, y_test


def load_SEED_V_cross_session_data(
    subject: int,
    session: int = 1,
    data_root: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train on sessions except `session`, test on `session` for one subject.
    """
    root = Path(data_root) if data_root else _default_seed_v_root()
    if session not in (1, 2, 3):
        raise ValueError("cross-session requires session in {1,2,3}.")

    labels, _ = get_seed_v_labels(root, session_id=session)
    x_test, y_test, _ = _collect_subject_session_data(root, subject, session, labels)
    x_train_parts, y_train_parts = [], []
    for ses in (1, 2, 3):
        if ses == session:
            continue
        labels, _ = get_seed_v_labels(root, session_id=ses)
        xs, ys, _ = _collect_subject_session_data(root, subject, ses, labels)
        x_train_parts.append(xs)
        y_train_parts.append(ys)
    train_x = np.concatenate(x_train_parts, axis=0)
    train_y = np.concatenate(y_train_parts, axis=0)
    return train_x, train_y, x_test, y_test


def get_official_trial_labels(
    dataset: str,
    data_root: str | Path | None = None,
    session_id: int | None = None,
) -> dict[str, Any]:
    ds = dataset.strip().lower()
    root = Path(data_root) if data_root else _default_data_root()
    if ds == "seed":
        labels = get_seed_labels(root / "SEED")
        split = get_within_subject_split("seed", total_trials=len(labels))
        return {"dataset": ds, "labels": labels, "within_split": split}
    if ds == "seed_iv":
        if session_id is None:
            raise ValueError("session_id is required for seed_iv.")
        labels = get_seed_iv_labels(root / "SEED-IV", session_id=session_id)
        split = get_within_subject_split("seed_iv", total_trials=len(labels))
        return {"dataset": ds, "labels": labels, "within_split": split}
    if ds == "seed_v":
        seed_v_root = Path(data_root) / "SEED-V" if data_root else _default_seed_v_root()
        labels, stimuli = get_seed_v_labels(seed_v_root)
        split = get_within_subject_split("seed_v", total_trials=len(labels))
        return {"dataset": ds, "labels": labels, "stimuli_order": stimuli, "within_split": split}
    raise ValueError("dataset must be one of: seed, seed_iv, seed_v")


__all__ = [
    "DataLoadError",
    "get_official_trial_labels",
    "load_SEED_V_cross_session_data",
    "load_SEED_V_cross_subject_data",
    "load_SEED_V_within_session_data",
]
