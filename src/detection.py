"""Physical-state detection from machine current. States: -1 unknown, 0 off, 1 panel, 2 started, 3 cutting."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from hmmlearn.hmm import GaussianHMM

STATE = {-1: "Unknown", 0: "Off", 1: "Panel", 2: "Started", 3: "Cutting"}
LABEL_ID = {-1: "Tidak diketahui", 0: "Mati", 1: "Panel", 2: "Di-start", 3: "Memotong"}   # for CSV outputs
EPS = 0.01  # A, avoids log(0)


def log_current(I):
    return np.log10(np.clip(np.asarray(I, float), 0, None) + EPS)


def to_amps(x):
    return 10 ** np.asarray(x) - EPS


def segments(valid):
    """Lengths of consecutive valid runs (the HMM chain is cut at data gaps)."""
    d = np.diff(np.r_[0, np.asarray(valid, int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def find_valleys(x, n=4, bins=400, sigma=3):
    """n largest peaks of the log(1+count) histogram and the valleys between them.
    Empty bins are padded on both edges so a peak at 0 A is still detected."""
    pad = 4 * sigma
    w = (x.max() - x.min()) / bins
    h, e = np.histogram(x, bins + 2 * pad, (x.min() - pad * w, x.max() + pad * w))
    c, hs = (e[:-1] + e[1:]) / 2, gaussian_filter1d(np.log1p(h), sigma, mode="constant")
    pk, p = find_peaks(hs, prominence=0.05 * hs.max())
    top = np.sort(pk[np.argsort(p["prominences"])[-n:]])
    return np.array([c[a + np.argmin(hs[a:b])] for a, b in zip(top[:-1], top[1:])]), c, hs, c[top]


def threshold(x, valleys, delta=0.05, hold=2):
    """A41: multi-level hysteresis threshold (Schmitt trigger) with a `hold`-sample debounce."""
    up, down, out = valleys + delta, valleys - delta, np.empty(len(x), np.int8)
    cur = cand = int(np.searchsorted(valleys, x[0])); cnt = 0
    for i, v in enumerate(x):
        tgt = (int(np.searchsorted(up, v)) if cur < len(valleys) and v > up[cur] else
               int(np.searchsorted(down, v)) if cur > 0 and v < down[cur - 1] else cur)
        if tgt == cur:
            cand, cnt = cur, 0
        else:
            cnt = cnt + 1 if tgt == cand else 1; cand = tgt
            if cnt >= hold:
                cur, cnt = tgt, 0
        out[i] = cur
    return out


def _rank(means):
    r = np.empty(len(means), int); r[np.argsort(np.ravel(means))] = np.arange(len(means)); return r


def gmm(x, k=4, x_fit=None, seed=0):
    """A42: GMM on log current; components sorted by mean -> states 0..k-1."""
    g = GaussianMixture(k, random_state=seed).fit((x if x_fit is None else x_fit).reshape(-1, 1))
    return _rank(g.means_)[g.predict(x.reshape(-1, 1))].astype(np.int8), g


def hmm(x, lengths, g, x_fit=None, len_fit=None, n_iter=30):
    """A43: Gaussian HMM initialised from GMM g, initial transition diagonal 0.99, Viterbi decoding."""
    k, o = len(g.means_), np.argsort(g.means_.ravel())
    h = GaussianHMM(k, "diag", n_iter=n_iter, tol=1e-3, init_params="", random_state=0)
    h.startprob_ = np.full(k, 1 / k)
    h.transmat_ = np.full((k, k), 0.01 / (k - 1)) + np.eye(k) * (0.99 - 0.01 / (k - 1))
    h.means_, h.covars_ = g.means_[o].reshape(k, 1), g.covariances_[o].reshape(k, 1)
    if x_fit is None:
        x_fit, len_fit = x, lengths
    h.fit(x_fit.reshape(-1, 1), len_fit)
    return _rank(h.means_)[h.predict(x.reshape(-1, 1), lengths)].astype(np.int8), h


def runs(s):
    """(value, length in samples, start index) of each consecutive run."""
    s = np.asarray(s); i = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    return s[i], np.diff(np.r_[i, len(s)]), i


def flickers(s, state=3, max_len=2):
    """Number of `state` runs no longer than max_len samples (10 s on a 5 s grid)."""
    v, d, _ = runs(s); return int(np.sum((v == state) & (d <= max_len)))
