"""评测指标：KL(p||q)（KDE 口径）/ MMD^2 / sliced W2。

方法论（FlowForge 验证过的口径）：
- KL(p_true || p_model) = E_p[log p_true] - E_p[log q_hat]，
  q_hat 为模型样本上的高斯 KDE（Silverman 带宽）。所有方法用**同一**
  KDE 协议与同一批 true 样本（common random numbers）=> 配对比较公平。
- KL 非负性用大样本断言（n>=2048，容忍 -1e-3；小样本 MC 噪声可致假负）。
- MMD^2 用无偏估计 + 中位数带宽；sliced W2 用确定性投影角（16 个）。
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import wasserstein_distance

from rectforge.core.errors import MetricError


def silverman_bandwidth(X: np.ndarray) -> float:
    """Silverman 规则：h = mean_std * (4n/(d+2))^{-1/(d+4)}。"""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    std = float(X.std(axis=0).mean())
    if std <= 0:
        raise MetricError("E501: zero-variance samples cannot set KDE bandwidth")
    return std * (4.0 * n / (d + 2.0)) ** (-1.0 / (d + 4.0))


def kde_log_density(train: np.ndarray, h: float, query: np.ndarray, chunk: int = 256) -> np.ndarray:
    """高斯 KDE 的 log 密度，分块防内存爆炸。"""
    train = np.asarray(train, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    n, d = train.shape
    out = np.empty(query.shape[0], dtype=np.float64)
    log_norm = -np.log(n) - 0.5 * d * np.log(2.0 * np.pi) - d * np.log(h)
    inv = 1.0 / (2.0 * h * h)
    for i in range(0, query.shape[0], chunk):
        q = query[i : i + chunk]
        d2 = ((q[:, None, :] - train[None, :, :]) ** 2).sum(axis=2)
        out[i : i + chunk] = logsumexp(-d2 * inv, axis=1) + log_norm
    return out


def kl_p_q(true_log_prob, x_true: np.ndarray, model_samples: np.ndarray) -> float:
    """KL(p_true || q_hat)。x_true 与 model_samples 长度可以不同。"""
    x_true = np.asarray(x_true, dtype=np.float64)
    model_samples = np.asarray(model_samples, dtype=np.float64)
    h = silverman_bandwidth(model_samples)
    logq = kde_log_density(model_samples, h, x_true)
    logp = np.asarray(true_log_prob(x_true), dtype=np.float64)
    if logp.shape != logq.shape:
        raise MetricError("E501: log_prob/log_q shape mismatch")
    return float(np.mean(logp - logq))


def _rbf_kernel(A: np.ndarray, B: np.ndarray, gamma: float) -> np.ndarray:
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2)
    return np.exp(-gamma * d2)


def _median_gamma(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.concatenate([X, Y], axis=0)
    m = min(Z.shape[0], 512)
    Zs = Z[:m]
    d2 = ((Zs[:, None, :] - Zs[None, :, :]) ** 2).sum(axis=2)
    iu = np.triu_indices(m, k=1)
    med = float(np.median(d2[iu]))
    return 1.0 / med if med > 1e-12 else 1.0


def mmd2(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """无偏 MMD^2（RBF，中位数带宽）。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if gamma is None:
        gamma = _median_gamma(X, Y)
    n, m = X.shape[0], Y.shape[0]
    if n < 2 or m < 2:
        raise MetricError("E501: MMD needs >=2 samples per set")
    kxx = _rbf_kernel(X, X, gamma)
    kyy = _rbf_kernel(Y, Y, gamma)
    kxy = _rbf_kernel(X, Y, gamma)
    term_xx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    term_yy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    term_xy = kxy.mean()
    return float(term_xx + term_yy - 2.0 * term_xy)


def sliced_w1(X: np.ndarray, Y: np.ndarray, n_proj: int = 16) -> float:
    """sliced Wasserstein-1：确定性投影角（黄金角序列）。"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    total = 0.0
    for k in range(n_proj):
        ang = np.pi * k * (3.0 - np.sqrt(5.0))  # 黄金角，确定性
        u = np.array([np.cos(ang), np.sin(ang)])
        total += wasserstein_distance(X @ u, Y @ u)
    return float(total / n_proj)
