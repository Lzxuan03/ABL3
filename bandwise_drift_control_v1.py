from __future__ import annotations
import numpy as np
from .cumulative_welford_v1 import WelfordState, CausalWindow, safe_var, safe_std
from .global_drift_control_v1 import sigmoid

BANDS = ['delta', 'theta', 'alpha', 'beta', 'gamma']
SHUFFLE_NEXT = np.asarray([1, 2, 3, 4, 0], dtype=np.int64)


def reshape_62x5(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 310:
            raise ValueError(f'cannot reshape feature of size {arr.size} to [62,5]')
        return arr.reshape(62, 5)
    if arr.shape[-2:] == (62, 5):
        return arr
    if arr.shape[-1] == 310:
        return arr.reshape(*arr.shape[:-1], 62, 5)
    raise ValueError(f'cannot reshape {arr.shape} to [62,5]')


def band_drift(cmean, cvar, rmean, rvar, eta=0.5, use_mean=True, use_var=True):
    cm = reshape_62x5(cmean); cv = reshape_62x5(safe_var(cvar))
    rm = reshape_62x5(rmean); rv = reshape_62x5(safe_var(rvar))
    d_mean = np.mean(np.abs(rm - cm) / (safe_std(cv) + 1e-12), axis=0)
    d_var = np.mean(np.abs(np.log(rv + 1e-12) - np.log(cv + 1e-12)), axis=0)
    if not use_mean:
        d_mean = np.zeros_like(d_mean)
    if not use_var:
        d_var = np.zeros_like(d_var)
    return d_mean + float(eta) * d_var, d_mean, d_var


def band_lambdas(drift_by_band, theta=1.0, temperature=0.25, lambda_min=0.98):
    d = np.asarray(drift_by_band, dtype=np.float64)
    return 1.0 - (1.0 - float(lambda_min)) * sigmoid((d - float(theta)) / max(float(temperature), 1e-6))


class BandwiseWelford:
    def __init__(self, channels=62, bands=5):
        self.channels = int(channels)
        self.bands = int(bands)
        self.states = [WelfordState(self.channels) for _ in range(self.bands)]

    def update(self, x, lambdas=None):
        xb = reshape_62x5(x)
        if lambdas is None:
            lambdas = np.ones(self.bands, dtype=np.float64)
        lambdas = np.asarray(lambdas, dtype=np.float64)
        for b in range(self.bands):
            self.states[b].update(xb[:, b], retention=float(lambdas[b]), update_weight=1.0)

    def stats(self, fallback_mean=None, fallback_var=None):
        fm = reshape_62x5(fallback_mean) if fallback_mean is not None else None
        fv = reshape_62x5(fallback_var) if fallback_var is not None else None
        means, vars_, counts = [], [], []
        for b, st in enumerate(self.states):
            m, v, w = st.stats(None if fm is None else fm[:, b], None if fv is None else fv[:, b])
            means.append(m); vars_.append(v); counts.append(w)
        mean = np.stack(means, axis=1).reshape(-1)
        var = np.stack(vars_, axis=1).reshape(-1)
        return mean, safe_var(var), np.asarray(counts, dtype=np.float64)


class BandwiseWindow:
    def __init__(self, size=64, channels=62, bands=5):
        self.win = CausalWindow(channels * bands, size)

    def update(self, x):
        self.win.update(np.asarray(x, dtype=np.float64).reshape(-1))

    def stats(self, fallback_mean=None, fallback_var=None):
        return self.win.stats(fallback_mean, fallback_var)
