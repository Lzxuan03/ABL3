from __future__ import annotations
import hashlib, json
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

EPS = 1e-5

def stable_hash_array(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(arr))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.view(np.uint8))
    return h.hexdigest()[:16]

def stable_hash_obj(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]

def safe_std(var: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(var, 0.0) + EPS)

def normalize_with_stats(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    return ((x - mean) / safe_std(var)).astype(np.float32)

@dataclass
class RunningStats:
    dim: int
    n: int = 0
    mean: Optional[np.ndarray] = None
    m2: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.mean is None:
            self.mean = np.zeros(self.dim, dtype=np.float64)
        if self.m2 is None:
            self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2

    def stats(self, fallback_mean: np.ndarray, fallback_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        if self.n < 2:
            return fallback_mean.astype(np.float64), fallback_var.astype(np.float64), self.n
        return self.mean.copy(), (self.m2 / max(1, self.n - 1)).copy(), self.n

class EMAStats:
    def __init__(self, dim: int, beta: float, init_mean: np.ndarray, init_var: np.ndarray):
        self.dim = dim
        self.beta = float(beta)
        self.n = 0
        self.mean = init_mean.astype(np.float64).copy()
        self.second = (init_var + init_mean ** 2).astype(np.float64).copy()

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1
        self.mean = (1.0 - self.beta) * self.mean + self.beta * x
        self.second = (1.0 - self.beta) * self.second + self.beta * (x ** 2)

    def stats(self) -> Tuple[np.ndarray, np.ndarray, int]:
        var = np.maximum(self.second - self.mean ** 2, 0.0)
        return self.mean.copy(), var.copy(), self.n

class WindowStats:
    def __init__(self, dim: int, window_size: int):
        self.dim = dim
        self.buf = deque(maxlen=int(window_size))

    @property
    def n(self):
        return len(self.buf)

    def update(self, x: np.ndarray):
        self.buf.append(np.asarray(x, dtype=np.float64).copy())

    def stats(self, fallback_mean: np.ndarray, fallback_var: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        if len(self.buf) < 2:
            return fallback_mean.astype(np.float64), fallback_var.astype(np.float64), len(self.buf)
        a = np.stack(list(self.buf), axis=0)
        return a.mean(0), a.var(0, ddof=1), len(self.buf)

def prefix_fixed_transform(x: np.ndarray, prefix_fraction: float) -> Tuple[np.ndarray, np.ndarray, dict]:
    n = x.shape[0]
    k = max(2, min(n - 1, int(round(n * float(prefix_fraction))))) if n > 2 else max(1, n)
    prefix = x[:k]
    mean = prefix.mean(0)
    var = prefix.var(0, ddof=1) if k > 1 else np.ones(x.shape[1], dtype=np.float32)
    z = normalize_with_stats(x[k:], mean, var)
    meta = {'prefix_sample_count': int(k), 'evaluation_sample_count': int(max(0, n-k)), 'prefix_hash': stable_hash_array(prefix), 'future_eval_hash': stable_hash_array(x[k:])}
    return z, np.arange(k, n, dtype=np.int64), meta

def source_prior(parts: List[dict], session_id: int | None = None) -> Tuple[np.ndarray, np.ndarray, dict]:
    chunks = []
    for p in parts:
        if session_id is None or int(p['session']) == int(session_id):
            chunks.append(np.asarray(p['eeg'], dtype=np.float32))
    if not chunks:
        for p in parts:
            chunks.append(np.asarray(p['eeg'], dtype=np.float32))
    x = np.concatenate(chunks, axis=0)
    mean = x.mean(0).astype(np.float64)
    var = x.var(0, ddof=1).astype(np.float64)
    return mean, var, {'sample_count': int(x.shape[0]), 'mean_hash': stable_hash_array(mean.astype(np.float32)), 'variance_hash': stable_hash_array(var.astype(np.float32))}

def shrink_stats(t_mean, t_var, n_eff, s_mean, s_var, tau: float):
    alpha = float(n_eff) / (float(n_eff) + float(tau)) if n_eff > 0 else 0.0
    mean = alpha * t_mean + (1 - alpha) * s_mean
    var = alpha * t_var + (1 - alpha) * s_var
    return mean, np.maximum(var, EPS), alpha

def prequential_transform(x: np.ndarray, method: str, *, warmup_count: int = 16, source_mean: np.ndarray, source_var: np.ndarray, tau: float = 256, beta: float = 0.01, window_size: int = 64, clip_k: float = 3.0, kappa: float = 0.1) -> Tuple[np.ndarray, np.ndarray, dict]:
    x = np.asarray(x, dtype=np.float64)
    dim = x.shape[1]
    run = RunningStats(dim)
    ema = EMAStats(dim, beta, source_mean, source_var)
    win = WindowStats(dim, window_size)
    zs, idxs, drift_scores, weights, alphas = [], [], [], [], []
    for i, xi in enumerate(x):
        r_mean, r_var, rn = run.stats(source_mean, source_var)
        w_mean, w_var, wn = win.stats(source_mean, source_var)
        if method == 'cumulative_welford':
            mean, var = r_mean, r_var
            alpha = rn / max(1, rn + tau)
            weight = (0.0, 1.0, 0.0)
        elif method == 'eb_shrinkage':
            mean, var, alpha = shrink_stats(r_mean, r_var, rn, source_mean, source_var, tau)
            weight = (1-alpha, alpha, 0.0)
        elif method == 'ema':
            mean, var, _ = ema.stats()
            alpha = min(1.0, ema.n / max(1.0, ema.n + tau))
            weight = (1-alpha, alpha, 0.0)
        elif method == 'sliding_window':
            mean, var = w_mean, w_var
            alpha = wn / max(1, wn + tau)
            weight = (1-alpha, 0.0, alpha)
        elif method == 'robust':
            mean, var = r_mean, r_var
            alpha = rn / max(1, rn + tau)
            weight = (1-alpha, alpha, 0.0)
        elif method in {'multiscale', 'analytic_reliability'}:
            drift = float(np.mean(np.abs(w_mean - r_mean)) + np.mean(np.abs(np.log(safe_std(w_var)) - np.log(safe_std(r_var)))))
            r_n = rn / max(1.0, rn + tau)
            r_s = float(np.exp(-drift / max(kappa, 1e-8)))
            w_source = 1.0 - r_n
            w_cum = r_n * r_s
            w_win = r_n * (1.0 - r_s)
            total = max(w_source + w_cum + w_win, 1e-12)
            w_source, w_cum, w_win = w_source/total, w_cum/total, w_win/total
            mean = w_source * source_mean + w_cum * r_mean + w_win * w_mean
            second = w_source * (source_var + source_mean**2) + w_cum * (r_var + r_mean**2) + w_win * (w_var + w_mean**2)
            var = np.maximum(second - mean**2, EPS)
            alpha = 1.0 - w_source
            weight = (w_source, w_cum, w_win)
            drift_scores.append(drift)
        else:
            raise ValueError(f'unknown method {method}')
        if i >= int(warmup_count):
            zs.append(normalize_with_stats(xi[None, :], mean, var)[0])
            idxs.append(i)
            weights.append(weight)
            alphas.append(alpha)
        update_x = xi
        if method == 'robust' and run.n >= 2:
            lo = r_mean - float(clip_k) * safe_std(r_var)
            hi = r_mean + float(clip_k) * safe_std(r_var)
            update_x = np.clip(xi, lo, hi)
        run.update(update_x)
        ema.update(update_x)
        win.update(update_x)
    meta = {
        'warmup_count': int(warmup_count),
        'evaluation_sample_count': int(len(idxs)),
        'evaluation_sample_hash': stable_hash_array(np.asarray(idxs, dtype=np.int64)),
        'drift_score_mean': float(np.mean(drift_scores)) if drift_scores else None,
        'alpha_mean': float(np.mean(alphas)) if alphas else None,
        'w_source_mean': float(np.mean([w[0] for w in weights])) if weights else None,
        'w_cumulative_mean': float(np.mean([w[1] for w in weights])) if weights else None,
        'w_window_mean': float(np.mean([w[2] for w in weights])) if weights else None,
    }
    return np.asarray(zs, dtype=np.float32), np.asarray(idxs, dtype=np.int64), meta

def future_invariance_check(x: np.ndarray, method: str, **kwargs) -> dict:
    if len(x) < 12:
        return {'future_invariance_passed': True, 'future_access_count': 0, 'max_abs_diff': 0.0, 'note': 'too_short'}
    kwargs = dict(kwargs)
    kwargs.pop('warmup_count', None)
    t = min(max(8, len(x)//3), len(x)-2)
    base, idx, _ = prequential_transform(x[:t+2], method, warmup_count=max(1, t-1), **kwargs)
    x2 = x[:t+2].copy()
    rng = np.random.default_rng(20260722)
    x2[t+1:] = rng.normal(size=x2[t+1:].shape)
    alt, idx2, _ = prequential_transform(x2, method, warmup_count=max(1, t-1), **kwargs)
    if len(base) == 0 or len(alt) == 0:
        diff = 0.0
    else:
        diff = float(np.max(np.abs(base[0] - alt[0])))
    return {'future_invariance_passed': bool(diff <= 1e-10), 'future_access_count': 0 if diff <= 1e-10 else 1, 'max_abs_diff': diff}
