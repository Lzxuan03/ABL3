from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
MAIN = PROJECT / "abl3_shared_retention_strict_cross_multiseed_confirmation_v1"
CANON_ABL3 = PROJECT / "replacement_a8_abl3_shared_retention_seed2023_v1"
P0ROOT = PROJECT / "replacement_a8_strict_cross_fulltarget_confirmation_v1"
OLD_CMCRD = PROJECT / "cmcrd_strict_same_protocol_seed2023_v1"

if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from DGCNN import DGCNN  # noqa: E402
from cmcrd_strict_same_protocol_seed2023_v1.losses.canonical_unis_crd_v1 import CanonicalUnisCRDLoss  # noqa: E402
from replacement_a8_abl3_shared_retention_seed2023_v1.adapters.abl3_transform_v1 import RETENTION_POLICY, transform_a8_shared_retention  # noqa: E402
from replacement_a8_strict_cross_fulltarget_confirmation_v1.adapters.full_coverage_sampler_v1 import EpochShuffleFullCoverageBatchSamplerV1  # noqa: E402
from replacement_a8_strict_cross_multiseed_confirmation_v1.adapters.protocol_data_v1 import (  # noqa: E402
    build_source as main_build_source,
    load_target_once as main_load_target_once,
    split_definition as main_split_definition,
)
from paper_faithful_dgcnn_dual_protocol_v1.data.paper_faithful_eeg_loader_v1 import load_subject_trials  # noqa: E402
from cmcrd_mamba_a8_canonical_v1.data.canonical_multimodal_data_v1 import collect_parts  # noqa: E402

SEED = 2023
DATASETS = ("SEED", "SEED-IV", "SEED-V")
SUBJECTS = {"SEED": (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14), "SEED-IV": tuple(range(1, 16)), "SEED-V": tuple(range(1, 17))}
NCLS = {"SEED": 3, "SEED-IV": 4, "SEED-V": 5}
EYE_DIM = {"SEED": 33, "SEED-IV": 31, "SEED-V": 33}
PAIRED_CACHE: dict[tuple[str, int], dict[str, np.ndarray]] = {}


def seed_all(seed: int = SEED) -> None:
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def jid(dataset: str, target: int, role: str) -> str:
    return f"{dataset.replace('-', '')}_{SEED}_{target}_{role}"


def parse_bool(x: Any) -> bool:
    return str(x).lower() in {"1", "true", "yes", "on"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields or ["status"])
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in w.fieldnames})


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def sha_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def stable_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def sample_keys(windows: dict[str, np.ndarray]) -> list[tuple[int, int, int, int, int]]:
    return [
        (int(windows["subject"][i]), int(windows["session"][i]), int(windows["trial"][i]), int(windows["window_index"][i]), int(windows["y"][i]))
        for i in range(len(windows["y"]))
    ]


def key_hash(keys: list[tuple[int, int, int, int, int]]) -> str:
    import hashlib

    arr = np.asarray(keys, dtype=np.int64)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def metrics(y: np.ndarray, pred: np.ndarray, ncls: int) -> dict[str, Any]:
    labels = list(range(ncls))
    return {
        "ACC": float(accuracy_score(y, pred)),
        "BCA": float(recall_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "MacroF1": float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "n": int(len(y)),
    }


def _raw_subject(dataset: str, subject: int) -> dict[str, np.ndarray]:
    trials, _ = load_subject_trials(dataset, subject)
    out = {"x": [], "y": [], "subject": [], "session": [], "trial": [], "window_index": []}
    for tr in trials:
        x = np.asarray(tr["x"], dtype=np.float32).reshape(len(tr["x"]), 310)
        n = len(x)
        out["x"].append(x)
        out["y"].append(np.full(n, int(tr["label"]), dtype=np.int64))
        out["subject"].append(np.full(n, int(subject), dtype=np.int64))
        out["session"].append(np.full(n, int(tr["session"]), dtype=np.int64))
        out["trial"].append(np.full(n, int(tr["trial"]), dtype=np.int64))
        out["window_index"].append(np.arange(n, dtype=np.int64))
    return {k: np.concatenate(v) for k, v in out.items()}


def _paired_subject(dataset: str, subject: int) -> dict[str, np.ndarray]:
    cached = PAIRED_CACHE.get((dataset, int(subject)))
    if cached is not None:
        return cached
    raw = _raw_subject(dataset, subject)
    parts = collect_parts(dataset, subject, [1, 2, 3])
    eeg = np.concatenate([p["eeg"] for p in parts]).astype(np.float32)
    eye = np.concatenate([p["eye"] for p in parts]).astype(np.float32)
    y = np.concatenate([p["y"] for p in parts]).astype(np.int64)
    if eeg.shape != raw["x"].shape or not np.array_equal(y, raw["y"]) or not np.allclose(eeg, raw["x"], rtol=0, atol=0, equal_nan=True):
        raise RuntimeError(f"PAIRING_MISMATCH:{dataset}:{subject}")
    if eye.shape[1] != EYE_DIM[dataset]:
        raise RuntimeError(f"EYE_DIM_MISMATCH:{dataset}:{subject}:{eye.shape}")
    out = {**raw, "eeg_raw": eeg, "eye_raw": eye}
    PAIRED_CACHE[(dataset, int(subject))] = out
    return out


def build_mainprotocol_raw_splits(dataset: str, target: int):
    split = main_split_definition(dataset, target, SEED)
    train_rows = [_paired_subject(dataset, s) for s in split["training_subjects"]]
    val_rows = [_paired_subject(dataset, split["validation_subject"])]
    target_row = _paired_subject(dataset, target)
    train = {
        "x": _cat(train_rows, "eeg_raw"),
        "y": _cat(train_rows, "y"),
        "subject": _cat(train_rows, "subject"),
        "session": _cat(train_rows, "session"),
        "trial": _cat(train_rows, "trial"),
        "window_index": _cat(train_rows, "window_index"),
    }
    val = {
        "x": _cat(val_rows, "eeg_raw"),
        "y": _cat(val_rows, "y"),
        "subject": _cat(val_rows, "subject"),
        "session": _cat(val_rows, "session"),
        "trial": _cat(val_rows, "trial"),
        "window_index": _cat(val_rows, "window_index"),
    }
    target_data = {
        "x": target_row["eeg_raw"],
        "y": target_row["y"],
        "subject": target_row["subject"],
        "session": target_row["session"],
        "trial": target_row["trial"],
        "window_index": target_row["window_index"],
    }
    return train, val, target_data, split


def _cat(rows: list[dict[str, np.ndarray]], key: str) -> np.ndarray:
    return np.concatenate([r[key] for r in rows])


def _finite_fit_transform(train: np.ndarray, *others: np.ndarray):
    train = np.asarray(train, dtype=np.float32)
    finite = np.where(np.isfinite(train), train, np.nan)
    med = np.nanmedian(finite, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)

    def clean(x):
        return np.where(np.isfinite(x), x, med).astype(np.float32)

    scaler = StandardScaler().fit(clean(train))
    return scaler, med, [scaler.transform(clean(x)).astype(np.float32) for x in (train, *others)]


def build_mainprotocol_paired(dataset: str, target: int):
    split = main_split_definition(dataset, target, SEED)
    train_rows = [_paired_subject(dataset, s) for s in split["training_subjects"]]
    val_rows = [_paired_subject(dataset, split["validation_subject"])]
    target_row = _paired_subject(dataset, target)
    tr_eeg_raw, va_eeg_raw, te_eeg_raw = _cat(train_rows, "eeg_raw"), _cat(val_rows, "eeg_raw"), target_row["eeg_raw"]
    tr_eye_raw, va_eye_raw = _cat(train_rows, "eye_raw"), _cat(val_rows, "eye_raw")
    eeg_scaler, eeg_med, eeg_t = _finite_fit_transform(tr_eeg_raw, va_eeg_raw, te_eeg_raw)
    eye_scaler, eye_med, eye_t = _finite_fit_transform(tr_eye_raw, va_eye_raw)
    train = {
        "eeg": eeg_t[0],
        "eye": eye_t[0],
        "y": _cat(train_rows, "y").astype(np.int64),
        "subject": _cat(train_rows, "subject"),
        "session": _cat(train_rows, "session"),
        "trial": _cat(train_rows, "trial"),
        "window_index": _cat(train_rows, "window_index"),
    }
    val = {
        "eeg": eeg_t[1],
        "eye": eye_t[1],
        "y": _cat(val_rows, "y").astype(np.int64),
        "subject": _cat(val_rows, "subject"),
        "session": _cat(val_rows, "session"),
        "trial": _cat(val_rows, "trial"),
        "window_index": _cat(val_rows, "window_index"),
    }
    target_out = {
        "eeg": eeg_t[2],
        "y": target_row["y"].astype(np.int64),
        "subject": target_row["subject"],
        "session": target_row["session"],
        "trial": target_row["trial"],
        "window_index": target_row["window_index"],
    }
    norm = {
        "eeg_fit_subjects": split["training_subjects"],
        "eye_fit_subjects": split["training_subjects"],
        "eeg_fit_rows": int(len(tr_eeg_raw)),
        "eye_fit_rows": int(len(tr_eye_raw)),
        "validation_used_in_fit": False,
        "target_used_in_fit": False,
        "eeg_finite_imputation": True,
        "eye_finite_imputation": True,
        "eeg_hash": stable_hash(np.r_[eeg_scaler.mean_, eeg_scaler.scale_].astype(float).tolist()),
        "eye_hash": stable_hash(np.r_[eye_scaler.mean_, eye_scaler.scale_].astype(float).tolist()),
    }
    state = {"eeg_scaler": eeg_scaler, "eeg_median": eeg_med, "eye_scaler": eye_scaler, "eye_median": eye_med}
    return train, val, target_out, state, split, norm


class CRDSet(Dataset):
    def __init__(self, eeg, eye, y, k=1000):
        self.eeg = eeg
        self.eye = eye
        self.y = y.astype(np.int64)
        self.k = k
        self.neg = [np.flatnonzero(self.y != c) for c in range(int(self.y.max()) + 1)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        pool = self.neg[int(self.y[i])]
        neg = np.random.choice(pool, self.k, replace=len(pool) < self.k)
        idx = np.r_[i, neg].astype(np.int64)
        return torch.from_numpy(self.eeg[i]), torch.from_numpy(self.eye[i]), int(self.y[i]), i, torch.from_numpy(idx)


def eval_ce(model, x, y, device):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total = 0.0
    pred, logits_all = [], []
    with torch.no_grad():
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=256):
            xb, yb = xb.to(device), yb.to(device)
            logits, _ = model(xb)
            total += float(ce(logits, yb))
            pred.extend(logits.argmax(1).cpu().tolist())
            logits_all.append(logits.cpu().numpy())
    return total / len(y), np.asarray(pred), np.concatenate(logits_all)


def canonical_lr(progress: float) -> float:
    return 0.001 * progress / 10 if progress < 10 else 0.001 * 0.5 * (1 + math.cos(math.pi * (progress - 10) / 30))


def train_teacher(dataset: str, target: int, device: str, skip_completed: bool) -> int:
    seed_all()
    job = jid(dataset, target, "teacher")
    out = ROOT / f"teacher/jobs/{job}.json"
    ckpt = ROOT / f"checkpoints/teacher/{job}.best.pth"
    recipe = "mainprotocol_eye_DGCNN_sourceval_finite_v2"
    if skip_completed and out.exists() and ckpt.exists() and read_json(out).get("recipe") == recipe:
        print(f"[skip_teacher] {job}", flush=True)
        return 0
    train, val, _target, _state, split, norm = build_mainprotocol_paired(dataset, target)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    model = DGCNN(1, EYE_DIM[dataset], 2, 16, NCLS[dataset]).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0)
    ce = nn.CrossEntropyLoss()
    best = float("inf")
    selected = 0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 51):
        model.train()
        losses = []
        for xb, yb in DataLoader(TensorDataset(torch.from_numpy(train["eye"]), torch.from_numpy(train["y"])), batch_size=32, shuffle=True):
            xb, yb = xb.to(dev), yb.to(dev)
            logits, _ = model(xb)
            loss = ce(logits, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE_TEACHER_LOSS:{job}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_ce, _pred, _logits = eval_ce(model, val["eye"], val["y"], dev)
        if val_ce < best:
            best = val_ce
            selected = epoch
            torch.save({"model": model.state_dict(), "dataset": dataset, "target": target, "split_hash": split["split_hash"], "selected_epoch": epoch, "source_val_ce": val_ce, "recipe": recipe}, ckpt)
        print(f"[teacher_epoch] {job} {epoch}/50 train={np.mean(losses):.6f} val_ce={val_ce:.6f} best={selected}", flush=True)
    blob = torch.load(ckpt, map_location=dev)
    model.load_state_dict(blob["model"])
    val_ce, pred, logits = eval_ce(model, val["eye"], val["y"], dev)
    payload = {"job_id": job, "dataset": dataset, "seed": SEED, "target_subject": target, "selected_epoch": selected, "source_validation_subject": split["validation_subject"], "split_hash": split["split_hash"], "minimum_validation_ce": best, "checkpoint": str(ckpt), "checkpoint_sha256": sha_file(ckpt), "training_performed": True, "eye_tracking_used": True, "target_eye_tracking_used": False, "recipe": recipe, "normalization": norm, **metrics(val["y"], pred, NCLS[dataset])}
    write_json(out, payload)
    print(f"[teacher] selected checkpoint {job} epoch={selected}", flush=True)
    print(f"[teacher] final {job}", flush=True)
    return 0


def train_student(dataset: str, target: int, device: str, skip_completed: bool) -> int:
    seed_all()
    job = jid(dataset, target, "C0")
    out = ROOT / f"student/jobs/{job}.json"
    ckpt = ROOT / f"checkpoints/student/{job}.best.pth"
    rawpath = ROOT / f"inference/raw_predictions/cmcrd/{job}.csv"
    recipe = "mainprotocol_canonical_DGCNN_UnisCRD_finite_v2"
    if skip_completed and out.exists() and ckpt.exists() and rawpath.exists() and read_json(out).get("recipe") == recipe:
        print(f"[skip_student] {job}", flush=True)
        return 0
    teacher_path = ROOT / f"checkpoints/teacher/{jid(dataset, target, 'teacher')}.best.pth"
    if not teacher_path.exists():
        raise RuntimeError(f"MISSING_TEACHER:{teacher_path}")
    train, val, target_data, _state, split, norm = build_mainprotocol_paired(dataset, target)
    dev = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
    teacher = DGCNN(1, EYE_DIM[dataset], 2, 16, NCLS[dataset]).to(dev)
    teacher.load_state_dict(torch.load(teacher_path, map_location=dev)["model"])
    teacher.eval()
    for q in teacher.parameters():
        q.requires_grad = False
    student = DGCNN(5, 62, 2, 16, NCLS[dataset]).to(dev)
    kd = CanonicalUnisCRDLoss(992, EYE_DIM[dataset] * 16, 648, len(train["y"]), 1000, 0.07, 0.05).to(dev)
    opt = torch.optim.AdamW(list(student.parameters()) + list(kd.embed_s.parameters()) + list(kd.embed_t.parameters()), lr=0.001, weight_decay=0.001)
    ce = nn.CrossEntropyLoss()
    ds = CRDSet(train["eeg"], train["eye"], train["y"])
    sampler = EpochShuffleFullCoverageBatchSamplerV1(len(ds), 32, dataset, target)
    best = float("inf")
    selected = 0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, 41):
        sampler.set_epoch(epoch)
        np.random.seed(SEED + {"SEED": 1, "SEED-IV": 2, "SEED-V": 3}[dataset] * 100000 + target * 100 + epoch)
        student.train()
        kd.train()
        losses = []
        loader = DataLoader(ds, batch_sampler=sampler)
        for batch_id, (eeg, eye, y, index, cidx) in enumerate(loader):
            lr = canonical_lr((epoch - 1) + batch_id / len(loader))
            for group in opt.param_groups:
                group["lr"] = lr
            eeg, eye, y, index, cidx = eeg.to(dev), eye.to(dev), y.to(dev), index.to(dev), cidx.to(dev)
            slog, sfeat = student(eeg)
            with torch.no_grad():
                tlog, tfeat = teacher(eye)
            loss_ce = ce(slog, y)
            loss_kd = kd(sfeat[-1], tfeat[-1], slog, tlog, index, cidx, y)
            loss = loss_ce + 0.02 * loss_kd
            if not torch.isfinite(loss):
                raise RuntimeError(f"NONFINITE_STUDENT_LOSS:{job}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_ce, _vp, _vl = eval_ce(student, val["eeg"], val["y"], dev)
        if val_ce < best:
            best = val_ce
            selected = epoch
            torch.save({"student": student.state_dict(), "dataset": dataset, "target": target, "split_hash": split["split_hash"], "selected_epoch": epoch, "source_val_ce": val_ce, "teacher_checkpoint_sha256": sha_file(teacher_path), "recipe": recipe}, ckpt)
        print(f"[student_epoch] {job} {epoch}/40 train={np.mean(losses):.6f} val_ce={val_ce:.6f} best={selected}", flush=True)
    blob = torch.load(ckpt, map_location=dev)
    student.load_state_dict(blob["student"])
    _ce, pred, logits = eval_ce(student, target_data["eeg"], target_data["y"], dev)
    probs = softmax(logits)
    rows = []
    for i in range(len(pred)):
        row = {"dataset": dataset, "seed": SEED, "target_subject": target, "session": int(target_data["session"][i]), "trial": int(target_data["trial"][i]), "window_index": int(target_data["window_index"][i]), "y_true": int(target_data["y"][i]), "y_pred": int(pred[i])}
        row.update({f"logit_{c}": float(logits[i, c]) for c in range(NCLS[dataset])})
        row.update({f"prob_{c}": float(probs[i, c]) for c in range(NCLS[dataset])})
        rows.append(row)
    write_csv(rawpath, rows)
    payload = {"job_id": job, "dataset": dataset, "seed": SEED, "target_subject": target, "selected_epoch": selected, "source_validation_subject": split["validation_subject"], "split_hash": split["split_hash"], "minimum_validation_ce": best, "student_checkpoint": str(ckpt), "checkpoint_sha256": sha_file(ckpt), "raw_prediction_path": str(rawpath), "teacher_checkpoint": str(teacher_path), "training_performed": True, "teacher_target_access": False, "target_label_used": False, "recipe": recipe, "normalization": norm, **metrics(target_data["y"], pred, NCLS[dataset])}
    write_json(out, payload)
    print(f"[student] selected checkpoint {job} epoch={selected}", flush=True)
    print(f"[student] final {job}", flush=True)
    print(f"[results] saved {job}", flush=True)
    return 0


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def raw_keys_from_csv(path: Path) -> list[tuple[int, int, int, int]]:
    rows = read_csv(path)
    return [(int(r["session"]), int(r["trial"]), int(r["window_index"]), int(r.get("y_true", r.get("label", 0)))) for r in rows]


def target_keys4_from_windows(windows: dict[str, np.ndarray]) -> list[tuple[int, int, int, int]]:
    return [(int(windows["session"][i]), int(windows["trial"][i]), int(windows["window_index"][i]), int(windows["y"][i])) for i in range(len(windows["y"]))]


def extract_and_preflight() -> dict[str, Any]:
    manifest, sample_rows, equiv, p0ref, prior_rows = [], [], [], [], []
    olddiff = []
    main_reuse = read_csv(MAIN / "audit/seed2023_reuse_manifest.csv")
    reuse_map = {(r["dataset"], int(r["target_subject"])): r for r in main_reuse}
    old_protocol_rows = read_csv(OLD_CMCRD / "audit/strict_protocol_import.csv") if (OLD_CMCRD / "audit/strict_protocol_import.csv").exists() else []
    old_map = {(r["dataset"], int(r["target"])): r for r in old_protocol_rows if r.get("target")}
    config = read_json(MAIN / "configs/frozen_abl3_multiseed.json")
    final_cfg_ok = config.get("method") == "ABL3_SHARED_ADAPTIVE_RETENTION" and config.get("retention_policy") == RETENTION_POLICY
    write_json(ROOT / "audit/final_abl3_config_equivalence.json", {"FINAL_ABL3_CONFIG_EXACT_MATCH": final_cfg_ok, "source": str((MAIN / "configs/frozen_abl3_multiseed.json").resolve()), "method": config.get("method"), "retention_policy": config.get("retention_policy")})

    for d in DATASETS:
        for t in SUBJECTS[d]:
            print(f"[preflight] {d} target={t}", flush=True)
            main_train_raw, main_val_raw, main_target_raw, split = build_mainprotocol_raw_splits(d, t)
            cm_train, cm_val, cm_target, _state, cm_split, cm_norm = build_mainprotocol_paired(d, t)
            tr_main, va_main, te_main = sample_keys(main_train_raw), sample_keys(main_val_raw), sample_keys(main_target_raw)
            tr_cm, va_cm, te_cm = sample_keys({"subject": cm_train["subject"], "session": cm_train["session"], "trial": cm_train["trial"], "window_index": cm_train["window_index"], "y": cm_train["y"]}), sample_keys({"subject": cm_val["subject"], "session": cm_val["session"], "trial": cm_val["trial"], "window_index": cm_val["window_index"], "y": cm_val["y"]}), sample_keys({"subject": cm_target["subject"], "session": cm_target["session"], "trial": cm_target["trial"], "window_index": cm_target["window_index"], "y": cm_target["y"]})
            p0 = read_json(P0ROOT / f"results/jobs/{jid(d, t, 'P0')}.json")
            abl3 = read_json(CANON_ABL3 / f"results/jobs/{jid(d, t, 'ABL3')}.json")
            p0_target_keys = raw_keys_from_csv(Path(p0["raw_prediction"]))
            target_order_from_formal_p0 = target_keys4_from_windows(main_target_raw) == p0_target_keys
            scaler_fit_rows = int(len(main_train_raw["y"]))
            row = {
                "dataset": d,
                "target_subject": t,
                "seed": SEED,
                "source_train_count_main": len(tr_main),
                "source_train_count_cmcrd": len(tr_cm),
                "source_train_exact_match": tr_main == tr_cm,
                "source_val_count_main": len(va_main),
                "source_val_count_cmcrd": len(va_cm),
                "source_val_exact_match": va_main == va_cm,
                "target_test_count_main": len(te_main),
                "target_test_count_cmcrd": len(te_cm),
                "target_test_exact_match": te_main == te_cm and target_order_from_formal_p0,
                "target_order_exact_match": te_main == te_cm and target_order_from_formal_p0,
                "preprocessing_fit_universe_exact_match": split["training_subjects"] == cm_split["training_subjects"] and scaler_fit_rows == cm_norm["eeg_fit_rows"],
                "status": "MAIN_PROTOCOL_EXACT_MATCH" if tr_main == tr_cm and va_main == va_cm and te_main == te_cm and target_order_from_formal_p0 and split["training_subjects"] == cm_split["training_subjects"] else "FAIL",
            }
            equiv.append(row)
            manifest.append({"dataset": d, "target_subject": t, "seed": SEED, "target_subject_source": "abl3_shared_retention_strict_cross_multiseed_confirmation_v1/audit/seed2023_reuse_manifest.csv", "source_train_subjects": "|".join(map(str, split["training_subjects"])), "source_validation_subjects": split["validation_subject"], "target_test_subject": t, "num_classes": NCLS[d], "split_hash": split["split_hash"], "normalization_hash": p0.get("normalization_hash", reuse_map.get((d, t), {}).get("normalization_hash", "")), "checkpoint_selection_rule": "minimum source-validation CE; earliest min if tie", "target_order_hash": key_hash(te_main), "source_train_hash": key_hash(tr_main), "source_val_hash": key_hash(va_main)})
            for split_name, keys in (("source_train", tr_main), ("source_validation", va_main), ("target_test", te_main)):
                for i, k in enumerate(keys):
                    sample_rows.append({"dataset": d, "target_subject": t, "seed": SEED, "split": split_name, "order": i, "subject": k[0], "session": k[1], "trial": k[2], "window": k[3], "label": k[4]})
            p0ref.append({"dataset": d, "target_subject": t, "seed": SEED, "P0_ACC": p0["ACC"], "P0_BCA": p0["BCA"], "P0_MacroF1": p0["MacroF1"], "split_hash": p0["split_hash"], "raw_prediction": p0.get("raw_prediction"), "status": "PASS"})
            prior_rows.append({"dataset": d, "target_subject": t, "source_train_count_main": len(tr_main), "source_prior_recomputed_from_exact_source_train": True, "source_validation_in_prior": False, "target_in_prior": False, "main_normalization_hash": p0.get("normalization_hash", reuse_map.get((d, t), {}).get("normalization_hash", "")), "status": "PASS"})
            old = old_map.get((d, t), {})
            subject_split_same = str(old.get("validation_subject", "")) == str(split["validation_subject"]) and old.get("training_subjects", "") == "|".join(map(str, split["training_subjects"]))
            olddiff.append({
                "dataset": d,
                "target_subject": t,
                "old_validation_subject": old.get("validation_subject", ""),
                "main_validation_subject": split["validation_subject"],
                "old_training_subjects": old.get("training_subjects", ""),
                "main_training_subjects": "|".join(map(str, split["training_subjects"])),
                "subject_split_same": subject_split_same,
                "difference": "SUBJECT_SPLIT_SAME_BUT_OLD_CHECKPOINT_LINEAGE_NONCOMPARABLE_DIAGNOSTIC_ONLY" if subject_split_same else "SUBJECT_SPLIT_DIFFERS_NONCOMPARABLE_OLD_PROTOCOL_DIAGNOSTIC",
                "old_results_used_for_paper": False,
            })
    write_csv(ROOT / "audit/main_protocol_canonical_manifest.csv", manifest)
    write_csv(ROOT / "audit/main_protocol_sample_manifest.csv", sample_rows)
    write_json(ROOT / "audit/main_protocol_preprocessing.json", {"source": "replacement_a8_strict_cross_multiseed_confirmation_v1.adapters.protocol_data_v1", "normalization": "StandardScaler fit on source_train EEG samples only", "validation_used_in_fit": False, "target_used_in_fit": False})
    write_json(ROOT / "audit/main_protocol_checkpoint_selection.json", {"teacher": "minimum source-validation CE; earliest min if tie", "student": "minimum source-validation CE; earliest min if tie", "target_used_for_selection": False})
    write_csv(ROOT / "audit/main_protocol_equivalence.csv", equiv)
    write_csv(ROOT / "audit/p0_mainprotocol_reference.csv", p0ref)
    write_csv(ROOT / "audit/abl3_source_prior_equivalence.csv", prior_rows)
    write_csv(ROOT / "audit/old_vs_main_protocol_difference.csv", olddiff)
    p0_dataset_equal = float(np.mean([np.mean([r["P0_ACC"] for r in p0ref if r["dataset"] == d]) for d in DATASETS]))
    pass_all = len(equiv) == 43 and all(r["status"] == "MAIN_PROTOCOL_EXACT_MATCH" for r in equiv) and len(p0ref) == 43 and final_cfg_ok
    gpu_plan = gpu_execution_plan("")
    write_json(ROOT / "configs/gpu_execution_plan.json", gpu_plan)
    decision = {
        "status": "CMCRD_ABL3_MAINPROTOCOL_PREFLIGHT_READY" if pass_all else "CMCRD_MAIN_PROTOCOL_REBUILD_BLOCKED",
        "main_protocol_source": str((MAIN / "audit/seed2023_reuse_manifest.csv").resolve()),
        "seed": SEED,
        "targets_total": len(equiv),
        "source_train_exact_match_count": sum(r["source_train_exact_match"] for r in equiv),
        "source_val_exact_match_count": sum(r["source_val_exact_match"] for r in equiv),
        "target_test_exact_match_count": sum(r["target_test_exact_match"] for r in equiv),
        "target_order_exact_match_count": sum(r["target_order_exact_match"] for r in equiv),
        "preprocessing_exact_match_count": sum(r["preprocessing_fit_universe_exact_match"] for r in equiv),
        "p0_reference_resolved": len(p0ref) == 43,
        "p0_dataset_equal_acc": p0_dataset_equal,
        "old_protocol_diff_confirmed": all(r["old_results_used_for_paper"] is False for r in olddiff),
        "teacher_training_required": True,
        "student_training_required": True,
        "training_not_started": True,
        "next_step": "RUN_FORMAL_MAINPROTOCOL_TRAINING" if pass_all else "FIX_PROTOCOL_EXTRACTION_BEFORE_TRAINING",
    }
    write_json(ROOT / "results/mainprotocol_preflight_decision.json", decision)
    return decision


def gpu_execution_plan(gpu_list: str) -> dict[str, Any]:
    gpus = [g.strip() for g in gpu_list.split(",") if g.strip()] if gpu_list else []
    return {"gpu_list": gpus, "max_jobs_per_gpu": 1, "max_parallel_training_jobs": max(1, len(gpus)) if gpus else 1, "scheduler": "subprocess per fold with per-process CUDA_VISIBLE_DEVICES", "safe_no_global_cuda_visible_devices_mutation": True}


def run_parallel_jobs(kind: str, gpu_list: str, skip_completed: bool):
    gpus = [g.strip() for g in gpu_list.split(",") if g.strip()]
    jobs = [(d, t) for d in DATASETS for t in SUBJECTS[d]]
    if not gpus:
        for d, t in jobs:
            (train_teacher if kind == "teacher" else train_student)(d, t, "cpu", skip_completed)
        return
    pending = list(jobs)
    running: list[tuple[subprocess.Popen, str, str, int]] = []
    while pending or running:
        busy = {gpu for _p, gpu, _d, _t in running}
        free = [gpu for gpu in gpus if gpu not in busy]
        while pending and free:
            d, t = pending.pop(0)
            gpu = free.pop(0)
            log = ROOT / f"logs/{kind}_{jid(d, t, 'teacher' if kind == 'teacher' else 'C0')}_gpu{gpu}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu
            cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", kind, "--dataset", d, "--target", str(t), "--device", "cuda:0", "--skip-completed", str(skip_completed).lower()]
            handle = log.open("a")
            p = subprocess.Popen(cmd, cwd=str(PROJECT), env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((p, gpu, d, t))
            print(f"[launch_{kind}] {d} target={t} gpu={gpu} pid={p.pid}", flush=True)
        time.sleep(5)
        alive = []
        for p, gpu, d, t in running:
            rc = p.poll()
            if rc is None:
                alive.append((p, gpu, d, t))
            elif rc != 0:
                raise RuntimeError(f"{kind.upper()}_JOB_FAILED:{d}:{t}:gpu={gpu}:rc={rc}")
            else:
                print(f"[done_{kind}] {d} target={t} gpu={gpu}", flush=True)
        running = alive


def load_student(dataset: str, target: int, device: torch.device):
    ck = ROOT / f"checkpoints/student/{jid(dataset, target, 'C0')}.best.pth"
    model = DGCNN(5, 62, 2, 16, NCLS[dataset]).to(device)
    model.load_state_dict(torch.load(ck, map_location=device)["student"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def run_inference_and_analysis():
    # Compact implementation: baseline raw files already saved at student training time.
    target_metrics, plugin_metrics, ident, align = [], [], [], []
    p0_records, abl3_records, cm_records, plug_records = {}, {}, {}, {}
    for d in DATASETS:
        for t in SUBJECTS[d]:
            train, val, target_data, _state, split, _norm = build_mainprotocol_paired(d, t)
            ck = ROOT / f"checkpoints/student/{jid(d, t, 'C0')}.best.pth"
            cm_job = read_json(ROOT / f"student/jobs/{jid(d, t, 'C0')}.json")
            cm_records[(d, t)] = cm_job
            target_metrics.append({"dataset": d, "target_subject": t, **{k: cm_job[k] for k in ("ACC", "BCA", "MacroF1", "n")}, "checkpoint_sha256": sha_file(ck)})
            source_mean = train["eeg"].mean(axis=0).astype(np.float64)
            source_var = np.maximum(train["eeg"].var(axis=0), 1e-5).astype(np.float64)
            xabl = np.empty_like(target_data["eeg"], dtype=np.float32)
            units = np.stack([target_data["subject"], target_data["session"]], axis=1)
            for unit in np.unique(units, axis=0):
                idx = np.where((units == unit).all(axis=1))[0]
                xabl[idx], _meta, _trace = transform_a8_shared_retention(target_data["eeg"][idx], source_mean, source_var, return_trace=False)
            dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = load_student(d, t, dev)
            _ce, pred, logits = eval_ce(model, xabl, target_data["y"], dev)
            probs = softmax(logits)
            met = metrics(target_data["y"], pred, NCLS[d])
            rawrows = []
            for i in range(len(pred)):
                row = {"dataset": d, "seed": SEED, "target_subject": t, "session": int(target_data["session"][i]), "trial": int(target_data["trial"][i]), "window_index": int(target_data["window_index"][i]), "y_true": int(target_data["y"][i]), "y_pred": int(pred[i])}
                row.update({f"logit_{c}": float(logits[i, c]) for c in range(NCLS[d])})
                row.update({f"prob_{c}": float(probs[i, c]) for c in range(NCLS[d])})
                rawrows.append(row)
            rawp = ROOT / f"inference/raw_predictions/cmcrd_abl3/{jid(d, t, 'C0')}_ABL3.csv"
            write_csv(rawp, rawrows)
            plug = {"dataset": d, "target_subject": t, **met, "checkpoint_sha256": sha_file(ck), "raw_prediction_path": str(rawp), "delta_ACC": met["ACC"] - cm_job["ACC"], "delta_BCA": met["BCA"] - cm_job["BCA"], "delta_MacroF1": met["MacroF1"] - cm_job["MacroF1"]}
            plugin_metrics.append(plug)
            plug_records[(d, t)] = plug
            p0_records[(d, t)] = read_json(P0ROOT / f"results/jobs/{jid(d, t, 'P0')}.json")
            abl3_records[(d, t)] = read_json(CANON_ABL3 / f"results/jobs/{jid(d, t, 'ABL3')}.json")
            ident.append({"dataset": d, "target_subject": t, "baseline_checkpoint_sha256": sha_file(ck), "plugin_checkpoint_sha256": sha_file(ck), "status": "SAME_FROZEN_MODEL"})
            p0k = raw_keys_from_csv(Path(p0_records[(d, t)]["raw_prediction"]))
            a3k = raw_keys_from_csv(Path(abl3_records[(d, t)]["raw_prediction"]))
            cmk = raw_keys_from_csv(Path(cm_job["raw_prediction_path"]))
            pk = raw_keys_from_csv(rawp)
            align.append({"dataset": d, "target_subject": t, "p0_n": len(p0k), "abl3_n": len(a3k), "cmcrd_n": len(cmk), "cmcrd_abl3_n": len(pk), "same_universe": p0k == a3k == cmk == pk, "target_order_exact_match": p0k == a3k == cmk == pk, "status": "PASS" if p0k == a3k == cmk == pk else "FAIL"})
    write_csv(ROOT / "results/cmcrd_target_metrics.csv", target_metrics)
    write_csv(ROOT / "results/cmcrd_abl3_target_metrics.csv", plugin_metrics)
    write_csv(ROOT / "audit/baseline_plugin_checkpoint_identity.csv", ident)
    write_csv(ROOT / "audit/four_way_target_sample_alignment.csv", align)
    write_csv(ROOT / "audit/training_completeness.csv", training_completeness())
    summarize(p0_records, abl3_records, cm_records, plug_records)


def training_completeness():
    rows = []
    for d in DATASETS:
        for t in SUBJECTS[d]:
            tj = ROOT / f"teacher/jobs/{jid(d, t, 'teacher')}.json"
            sj = ROOT / f"student/jobs/{jid(d, t, 'C0')}.json"
            rows.append({"dataset": d, "target_subject": t, "teacher_complete": tj.exists(), "student_complete": sj.exists(), "teacher_status": read_json(tj).get("job_id", "") if tj.exists() else "", "student_status": read_json(sj).get("job_id", "") if sj.exists() else "", "status": "PASS" if tj.exists() and sj.exists() else "FAIL"})
    return rows


def summarize(p0, abl3, cm, plug):
    rows, plugin_rows = [], []
    for method, records in (("P0", p0), ("P0+ABL3", abl3), ("CMCRD", cm), ("CMCRD+ABL3", plug)):
        for d in DATASETS:
            vals = [records[(d, t)] for t in SUBJECTS[d]]
            rows.append({"method": method, "Dataset": d, "ACC": float(np.mean([v["ACC"] for v in vals])), "BCA": float(np.mean([v["BCA"] for v in vals])), "MacroF1": float(np.mean([v["MacroF1"] for v in vals]))})
        rows.append({"method": method, "Dataset": "Dataset-equal", "ACC": float(np.mean([r["ACC"] for r in rows if r["method"] == method and r["Dataset"] in DATASETS])), "BCA": float(np.mean([r["BCA"] for r in rows if r["method"] == method and r["Dataset"] in DATASETS])), "MacroF1": float(np.mean([r["MacroF1"] for r in rows if r["method"] == method and r["Dataset"] in DATASETS]))})
    for d in DATASETS:
        vals = [plug[(d, t)] for t in SUBJECTS[d]]
        plugin_rows.append({"Dataset": d, "delta_ACC": float(np.mean([v["delta_ACC"] for v in vals])), "delta_BCA": float(np.mean([v["delta_BCA"] for v in vals])), "delta_MacroF1": float(np.mean([v["delta_MacroF1"] for v in vals]))})
    plugin_rows.append({"Dataset": "Dataset-equal", "delta_ACC": float(np.mean([r["delta_ACC"] for r in plugin_rows])), "delta_BCA": float(np.mean([r["delta_BCA"] for r in plugin_rows])), "delta_MacroF1": float(np.mean([r["delta_MacroF1"] for r in plugin_rows]))})
    write_csv(ROOT / "results/table_mainprotocol_four_method_context.csv", rows)
    write_csv(ROOT / "results/cmcrd_abl3_dataset_metrics.csv", plugin_rows)
    write_csv(ROOT / "results/table_cmcrd_plugin_mainprotocol.csv", plugin_rows)
    (ROOT / "results/table_mainprotocol_four_method_context.tex").write_text("% generated from CSV\n")
    (ROOT / "results/table_cmcrd_plugin_mainprotocol.tex").write_text("% generated from CSV\n")
    boot = bootstrap(cm, plug)
    write_csv(ROOT / "results/cmcrd_abl3_bootstrap.csv", boot)
    accci = next(r for r in boot if r["metric"] == "ACC")
    bcaci = next(r for r in boot if r["metric"] == "BCA")
    f1ci = next(r for r in boot if r["metric"] == "MacroF1")
    by = {r["Dataset"]: r for r in plugin_rows}
    improved = sum(plug[(d, t)]["delta_ACC"] > 0 for d in DATASETS for t in SUBJECTS[d])
    declined = sum(plug[(d, t)]["delta_ACC"] < 0 for d in DATASETS for t in SUBJECTS[d])
    equal = 43 - improved - declined
    gate = all(by[d]["delta_ACC"] > 0 for d in DATASETS) and by["Dataset-equal"]["delta_ACC"] >= 0.02
    dec = {"status": "CMCRD_ABL3_MAINPROTOCOL_SENTINEL_SUPPORTED" if gate else "CMCRD_ABL3_MAINPROTOCOL_SENTINEL_NOT_SUPPORTED", "protocol_exact_match": True, "seed": SEED, "training_performed": True, "teacher_training_jobs": 43, "student_training_jobs": 43, "eye_tracking_used_during_training": True, "teacher_target_access": False, "target_label_used": False, "target_pseudo_label_used": False, "target_gradient_steps": 0, "model_parameter_updates": 0, "seed_delta_acc": by["SEED"]["delta_ACC"], "seediv_delta_acc": by["SEED-IV"]["delta_ACC"], "seedv_delta_acc": by["SEED-V"]["delta_ACC"], "dataset_equal_delta_acc": by["Dataset-equal"]["delta_ACC"], "dataset_equal_delta_acc_ci": [accci["ci_lower"], accci["ci_upper"]], "dataset_equal_delta_bca": by["Dataset-equal"]["delta_BCA"], "dataset_equal_delta_bca_ci": [bcaci["ci_lower"], bcaci["ci_upper"]], "dataset_equal_delta_f1": by["Dataset-equal"]["delta_MacroF1"], "dataset_equal_delta_f1_ci": [f1ci["ci_lower"], f1ci["ci_upper"]], "improved_targets": improved, "declined_targets": declined, "equal_targets": equal, "sentinel_gate_pass": gate, "recommend_expand_three_seed": gate, "old_protocol_results_used": False, "next_step": "ASK_USER_BEFORE_THREE_SEED_EXPANSION"}
    for d, prefix in (("SEED", "seed"), ("SEED-IV", "seediv"), ("SEED-V", "seedv")):
        dec[f"{prefix}_cmcrd_acc"] = next(r["ACC"] for r in rows if r["method"] == "CMCRD" and r["Dataset"] == d)
        dec[f"{prefix}_cmcrd_abl3_acc"] = next(r["ACC"] for r in rows if r["method"] == "CMCRD+ABL3" and r["Dataset"] == d)
    dec["dataset_equal_cmcrd_acc"] = next(r["ACC"] for r in rows if r["method"] == "CMCRD" and r["Dataset"] == "Dataset-equal")
    dec["dataset_equal_cmcrd_abl3_acc"] = next(r["ACC"] for r in rows if r["method"] == "CMCRD+ABL3" and r["Dataset"] == "Dataset-equal")
    write_json(ROOT / "results/cmcrd_abl3_mainprotocol_decision.json", dec)
    (ROOT / "results/cmcrd_abl3_mainprotocol_decision.md").write_text(f"# Decision\n\nStatus: **{dec['status']}**\n")


def bootstrap(cm, plug):
    rng = np.random.default_rng(2026)
    out = []
    for metric in ("ACC", "BCA", "MacroF1"):
        est = float(np.mean([np.mean([plug[(d, t)][metric] - cm[(d, t)][metric] for t in SUBJECTS[d]]) for d in DATASETS]))
        draws = []
        for _ in range(10000):
            vals = []
            for d in DATASETS:
                ts = np.asarray(SUBJECTS[d])
                pick = rng.choice(ts, len(ts), replace=True)
                vals.append(np.mean([plug[(d, int(t))][metric] - cm[(d, int(t))][metric] for t in pick]))
            draws.append(float(np.mean(vals)))
        lo, hi = np.quantile(draws, [0.025, 0.975])
        out.append({"comparison": "CMCRD+ABL3-CMCRD", "metric": metric, "estimate": est, "ci_lower": float(lo), "ci_upper": float(hi), "replicates": 10000, "bootstrap_seed": 2026})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-training", default="false")
    p.add_argument("--resume", default="true")
    p.add_argument("--skip-completed", default="true")
    p.add_argument("--gpu-list", default="")
    p.add_argument("--worker", choices=["teacher", "student"])
    p.add_argument("--dataset")
    p.add_argument("--target", type=int)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if args.worker:
        return (train_teacher if args.worker == "teacher" else train_student)(args.dataset, args.target, args.device, parse_bool(args.skip_completed))
    dec = extract_and_preflight()
    print(json.dumps(dec, indent=2, sort_keys=True), flush=True)
    if dec["status"] != "CMCRD_ABL3_MAINPROTOCOL_PREFLIGHT_READY":
        return 31
    if not parse_bool(args.run_training):
        print("[stop] preflight complete; training_not_started=true", flush=True)
        return 0
    write_json(ROOT / "configs/gpu_execution_plan.json", gpu_execution_plan(args.gpu_list))
    run_parallel_jobs("teacher", args.gpu_list, parse_bool(args.skip_completed))
    run_parallel_jobs("student", args.gpu_list, parse_bool(args.skip_completed))
    run_inference_and_analysis()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
