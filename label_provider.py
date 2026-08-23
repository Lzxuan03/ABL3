from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


class LabelFileError(RuntimeError):
    """Raised when official label files are missing or unparsable."""


def _resolve_file(root: str | Path, candidates: Iterable[str]) -> Path:
    root_path = Path(root)
    if root_path.is_file():
        return root_path
    if not root_path.exists():
        raise LabelFileError(
            f"Dataset root not found: {root_path}. "
            "Please place official dataset files under this path."
        )

    for name in candidates:
        direct = root_path / name
        if direct.exists():
            return direct
        recursive = list(root_path.rglob(name))
        if recursive:
            return recursive[0]

    expected = ", ".join(candidates)
    raise LabelFileError(
        f"Official label file not found under: {root_path}. "
        f"Expected one of: {expected}."
    )


def _to_zero_based(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).astype(int).reshape(-1)
    unique = sorted(np.unique(labels).tolist())
    mapping = {value: idx for idx, value in enumerate(unique)}
    return np.asarray([mapping[v] for v in labels], dtype=np.int64)


def _read_text_with_fallback(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise LabelFileError(
        f"Unable to decode text file: {path}. "
        "Tried utf-8/utf-8-sig/gbk/latin-1."
    )


def get_seed_labels(seed_root: str | Path) -> np.ndarray:
    """Read SEED official label.mat and return zero-based trial labels."""
    label_file = _resolve_file(seed_root, ("label.mat",))
    mat = loadmat(label_file)

    arrays = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        arr = np.asarray(value)
        if arr.size >= 15 and np.issubdtype(arr.dtype, np.number):
            arrays.append((key, arr.reshape(-1)))

    if not arrays:
        raise LabelFileError(
            f"Could not find numeric label array in {label_file}. "
            f"Available keys: {list(mat.keys())}"
        )

    key, labels = min(arrays, key=lambda item: item[1].size)
    labels = labels[:15]
    labels = _to_zero_based(labels)
    print(f"[label_provider] SEED labels loaded from key='{key}' in {label_file}")
    return labels


def get_seed_iv_labels(seed_iv_root: str | Path, session_id: int) -> np.ndarray:
    """Parse SEED-IV ReadMe.txt and return 24 trial labels for a session."""
    if session_id not in (1, 2, 3):
        raise ValueError(f"session_id must be 1/2/3, got {session_id}")

    readme = _resolve_file(
        seed_iv_root, ("ReadMe.txt", "README.txt", "readme.txt")
    )
    text = _read_text_with_fallback(readme)

    session_block_pattern = re.compile(
        r"(session\s*" + str(session_id) + r".*?)(?=session\s*[123]|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    block_match = session_block_pattern.search(text)
    block = block_match.group(1) if block_match else text

    candidates = []
    bracket_lists = re.findall(r"\[(.*?)\]", block, flags=re.DOTALL)
    for item in bracket_lists:
        ints = [int(x) for x in re.findall(r"-?\d+", item)]
        if len(ints) >= 24:
            candidates.append(ints[:24])

    if not candidates:
        for line in block.splitlines():
            if "label" not in line.lower():
                continue
            ints = [int(x) for x in re.findall(r"-?\d+", line)]
            if len(ints) >= 24:
                candidates.append(ints[:24])

    if not candidates:
        ints = [int(x) for x in re.findall(r"-?\d+", block)]
        if len(ints) >= 24:
            candidates.append(ints[:24])

    if not candidates:
        raise LabelFileError(
            f"Could not parse 24 labels for SEED-IV session {session_id} from {readme}."
        )

    labels = _to_zero_based(np.asarray(candidates[0], dtype=np.int64))
    print(f"[label_provider] SEED-IV session={session_id} labels loaded from {readme}")
    return labels


def _parse_seed_v_official_layout(
    xlsx: Path, sheets: dict[str, pd.DataFrame], session_id: int | None
) -> tuple[np.ndarray, list[str], str] | None:
    for sheet_name in sheets:
        raw = pd.read_excel(xlsx, sheet_name=sheet_name, header=None)
        if raw.empty:
            continue

        emotion_to_label = {}
        for _, row in raw.iterrows():
            values = [v for v in row.tolist() if not pd.isna(v)]
            if len(values) < 2:
                continue
            for idx, value in enumerate(values[:-1]):
                next_value = values[idx + 1]
                numeric = pd.to_numeric(pd.Series([next_value]), errors="coerce").iloc[0]
                if pd.isna(numeric):
                    continue
                name = str(value).strip()
                if name and not re.fullmatch(r"-?\d+(\.\d+)?", name):
                    emotion_to_label[name.lower()] = int(numeric)

        session_rows = {}
        for _, row in raw.iterrows():
            values = [v for v in row.tolist() if not pd.isna(v)]
            for idx, value in enumerate(values):
                match = re.fullmatch(r"session\s*([123])", str(value).strip(), flags=re.IGNORECASE)
                if not match:
                    continue
                sid = int(match.group(1))
                session_rows[sid] = [str(v).strip() for v in values[idx + 1 :] if str(v).strip()]

        sid = session_id if session_id in (1, 2, 3) else sorted(session_rows)[0] if session_rows else None
        if sid is None or sid not in session_rows:
            continue

        stimuli_order = session_rows[sid][:15]
        if len(stimuli_order) < 15:
            continue

        labels = []
        for item in stimuli_order:
            key = item.lower()
            if key in emotion_to_label:
                labels.append(emotion_to_label[key])
                continue
            numeric = pd.to_numeric(pd.Series([item]), errors="coerce").iloc[0]
            if not pd.isna(numeric):
                labels.append(int(numeric))

        if len(labels) >= 15:
            return _to_zero_based(np.asarray(labels[:15], dtype=np.int64)), stimuli_order, sheet_name

    return None


def get_seed_v_labels(
    seed_v_root: str | Path, session_id: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """
    Read SEED-V official emotion_label_and_stimuli_order.xlsx.
    Returns:
        labels: zero-based 15 trial labels
        stimuli_order: ordered stimuli/video names (length 15 when available)
    """
    if session_id is not None and session_id not in (1, 2, 3):
        raise ValueError(f"session_id must be 1/2/3 when provided, got {session_id}")

    xlsx = _resolve_file(seed_v_root, ("emotion_label_and_stimuli_order.xlsx",))
    sheets = pd.read_excel(xlsx, sheet_name=None)
    if not sheets:
        raise LabelFileError(f"No worksheet found in {xlsx}.")

    label_series = None
    stim_series = None
    label_sheet_name = None

    for sheet_name, df in sheets.items():
        cols = {str(c).strip().lower(): c for c in df.columns}
        label_col = None
        for name in cols:
            if "emotion" in name and "label" in name:
                label_col = cols[name]
                break
            if name in ("label", "emotion"):
                label_col = cols[name]
                break

        if label_col is None:
            for col in df.columns:
                numeric = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(numeric) >= 15 and numeric.nunique() <= 8:
                    label_col = col
                    break

        if label_col is None:
            continue

        numeric_labels = pd.to_numeric(df[label_col], errors="coerce").dropna()
        if len(numeric_labels) < 15:
            continue

        label_series = numeric_labels.iloc[:15].astype(int)
        label_sheet_name = sheet_name

        stim_col = None
        for name in cols:
            if "stimuli" in name or "video" in name or "film" in name or "order" in name:
                if cols[name] != label_col:
                    stim_col = cols[name]
                    break
        if stim_col is None:
            for col in df.columns:
                if col == label_col:
                    continue
                non_na = df[col].dropna()
                if len(non_na) >= 15 and not np.issubdtype(
                    pd.Series(non_na).dtype, np.number
                ):
                    stim_col = col
                    break

        if stim_col is not None:
            stim_series = df[stim_col].dropna().astype(str).iloc[:15]
        break

    if label_series is None:
        parsed = _parse_seed_v_official_layout(xlsx, sheets, session_id)
        if parsed is not None:
            labels, stimuli_order, label_sheet_name = parsed
            session_text = f", session={session_id}" if session_id else ""
            print(
                "[label_provider] SEED-V labels loaded from "
                f"{xlsx} (sheet='{label_sheet_name}'{session_text})."
            )
            return labels, stimuli_order

        available = {sheet: [str(c) for c in df.columns] for sheet, df in sheets.items()}
        raise LabelFileError(
            f"Failed to parse labels from {xlsx}. Available sheets/columns: {available}"
        )

    labels = _to_zero_based(label_series.to_numpy(dtype=np.int64))
    stimuli_order = stim_series.tolist() if stim_series is not None else []
    print(
        "[label_provider] SEED-V labels loaded from "
        f"{xlsx} (sheet='{label_sheet_name}')."
    )
    return labels, stimuli_order
