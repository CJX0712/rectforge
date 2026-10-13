"""合成密度：全部带解析 log p + 可复现采样器 + 归一化自检。

FlowForge 铁律（本项目最重要的一条）：
每个合成密度必须配「解析 log p 真值」+ 数值积分自检，且 sample() 与
log_prob() 的分布参数严格同源——否则 KL 估计会出现静默偏差（FlowForge
实测 sample 用 var=1.6 而 log_prob 按单位方差算，KL 估成负值）。
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from rectforge.core.errors import DensitySpecError, NormalizationError

_SQRT_2PI = np.sqrt(2.0 * np.pi)


class BaseDensity:
    """密度基类：2D、square bounds、解析 log p、显式 rng 采样。"""

    name = "base"
    dim = 2
    bounds = (-4.0, 4.0)
    norm_tol = 1e-6

    # ---- 接口 -------------------------------------------------------
    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        raise NotImplementedError

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    # ---- 归一化自检 -------------------------------------------------
    def normalization_error(self, grid: int = 801) -> float:
        """2D 复合梯形积分 vs 1.0（平滑密度谱精度；含 -inf 的密度置 0）。"""
        lo, hi = self.bounds
        xs = np.linspace(lo, hi, grid)
        gx, gy = np.meshgrid(xs, xs)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        lp = np.asarray(self.log_prob(pts), dtype=np.float64).reshape(grid, grid)
        if np.any(np.isnan(lp)):
            raise NormalizationError("E102: log_prob returned NaN on grid")
        with np.errstate(over="ignore", invalid="ignore"):
            p = np.where(np.isfinite(lp), np.exp(lp), 0.0)
        if not np.all(np.isfinite(p)):
            raise NormalizationError("E102: density overflows grid (log p too large)")
        area = float(np.trapezoid(np.trapezoid(p, xs, axis=1), xs))
        return abs(area - 1.0)

    def check_normalization(self, grid: int = 801) -> float:
        err = self.normalization_error(grid)
        if err > self.norm_tol:
            raise NormalizationError(
                f"E102: density {self.name} normalization error {err:.3e} > tol {self.norm_tol:.1e}"
            )
        return err


class GaussianTransfer(BaseDensity):
    """N(mu, cov)：用于闭式速度场/往返矩的正确性金标准（非基准 DGP）。"""

    def __init__(self, mu=(2.0, 0.0), cov=((1.0, 0.0), (0.0, 0.25))):
        self.name = "gauss2gauss"
        self.mu = np.asarray(mu, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.dim = self.mu.shape[0]
        self._chol = np.linalg.cholesky(self.cov)
        # bounds 必须覆盖均值 ±5σ（右尾外质量 ~3e-7），否则归一化自检报截断误差
        span = 5.0 * float(np.sqrt(np.max(np.diag(self.cov))))
        center = float(self.mu[0])
        self.bounds = (center - span - 1.0, center + span + 1.0)

    def sample(self, rng, n):
        z = rng.normal(size=(n, self.dim))
        return self.mu[None, :] + z @ self._chol.T

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        d = self.dim
        z = np.linalg.solve(self._chol, (x - self.mu[None, :]).T).T
        quad = np.einsum("ij,ij->i", z, z)
        logdet = 2.0 * np.sum(np.log(np.diag(self._chol)))
        return -0.5 * (quad + d * np.log(2.0 * np.pi) + logdet)


def gaussian_transfer_velocity(x, t, m0, s0, m1, s1):
    """线性路径下高斯->高斯的解析边缘速度 b_t(x)（独立耦合）。

    x_t = (1-t)x_0 + t x_1, (x_0,x_1) 独立高斯 =>
    x_t ~ N(m_t, S_t), m_t=(1-t)m0+t m1, S_t=(1-t)^2 S0 + t^2 S1,
    Cov(x1-x0, x_t) = -(1-t)S0 + t S1。
    """
    m0 = np.asarray(m0, float)
    s0 = np.asarray(s0, float)
    m1 = np.asarray(m1, float)
    s1 = np.asarray(s1, float)
    x = np.atleast_2d(np.asarray(x, float))
    mt = (1.0 - t) * m0 + t * m1
    st = (1.0 - t) ** 2 * s0 + t**2 * s1
    cross = -(1.0 - t) * s0 + t * s1
    solve = np.linalg.solve(st, (x - mt[None, :]).T).T
    return (m1 - m0)[None, :] + solve @ cross.T


class GMM(BaseDensity):
    """等权/加权对角高斯混合。"""

    def __init__(self, name, means, stds, weights=None, bounds=(-3.6, 3.6)):
        self.name = name
        self.means = np.asarray(means, dtype=np.float64)
        self.stds = np.asarray(stds, dtype=np.float64)
        k = self.means.shape[0]
        self.weights = np.full(k, 1.0 / k) if weights is None else np.asarray(weights, float)
        self.weights = self.weights / self.weights.sum()
        self.dim = self.means.shape[1]
        self.bounds = bounds
        self._logw = np.log(self.weights)

    def sample(self, rng, n):
        comp = rng.choice(self.means.shape[0], size=n, p=self.weights)
        z = rng.normal(size=(n, self.dim))
        return self.means[comp] + z * self.stds[comp]

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        d2 = ((x[:, None, :] - self.means[None, :, :]) / self.stds[None, :, :]) ** 2
        log_norm_k = np.log(self.stds).sum(axis=1)  # (k,) 每分量各自的归一化常数
        comp_lp = -0.5 * d2.sum(axis=2) - log_norm_k[None, :] - 0.5 * self.dim * np.log(2.0 * np.pi)
        return logsumexp(self._logw[None, :] + comp_lp, axis=1)


class Ring(BaseDensity):
    """细圆环：theta ~ U(0,2pi), r ~ N(R, s)。解析密度含 1/r 极坐标因子。"""

    def __init__(self, name="ring", radius=2.0, radial_std=0.18, bounds=(-3.2, 3.2)):
        self.name = name
        self.radius = float(radius)
        self.radial_std = float(radial_std)
        self.dim = 2
        self.bounds = bounds

    def sample(self, rng, n):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
        r = self.radius + self.radial_std * rng.normal(size=n)
        return np.column_stack([r * np.cos(theta), r * np.sin(theta)])

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        r = np.hypot(x[:, 0], x[:, 1])
        with np.errstate(divide="ignore"):
            lp = (
                -0.5 * ((r - self.radius) / self.radial_std) ** 2
                - np.log(self.radial_std)
                - 0.5 * np.log(2.0 * np.pi)
                - np.log(2.0 * np.pi)
                - np.log(r)
            )
        return np.where(r > 1e-12, lp, -np.inf)


class Wave(BaseDensity):
    """非线性相关（正弦）密度：x1 ~ N(0,1)，x2|x1 ~ N(a sin(b x1), s)。"""

    def __init__(self, name="wave", amp=0.8, freq=2.0, cond_std=0.25, bounds=(-4.5, 4.5)):
        self.name = name
        self.amp = float(amp)
        self.freq = float(freq)
        self.cond_std = float(cond_std)
        self.dim = 2
        self.bounds = bounds
        self.norm_tol = 2e-5

    def sample(self, rng, n):
        x1 = rng.normal(size=n)
        x2 = self.amp * np.sin(self.freq * x1) + self.cond_std * rng.normal(size=n)
        return np.column_stack([x1, x2])

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        m = self.amp * np.sin(self.freq * x[:, 0])
        lp1 = -0.5 * x[:, 0] ** 2 - 0.5 * np.log(2.0 * np.pi)
        lp2 = -0.5 * ((x[:, 1] - m) / self.cond_std) ** 2 - np.log(self.cond_std) - 0.5 * np.log(2.0 * np.pi)
        return lp1 + lp2


class SinusoidalGrid(BaseDensity):
    """正弦棋盘格：p(x) ∝ (1+a cos(2πx/p))(1+a cos(2πy/p))，盒内紧支撑。

    平滑（1+a cos >= 1-a > 0）=> 梯形积分谱精度；多峰网格结构对
    ODE 采样的路径规划是真实挑战（相对高斯混合更贴近图像流应用）。
    """

    def __init__(self, name="checker", pitch=1.6, amp=0.85, bounds=(-0.95, 0.95)):
        self.name = name
        self.pitch = float(pitch)
        self.amp = float(amp)
        self.dim = 2
        self.bounds = bounds
        # 盒界不连续 => 复合梯形误差 O(grid spacing)（实测 ~3e-4 @801 点），
        # 属求积性质而非密度错误；容差放到 2e-3 并如实记录。
        self.norm_tol = 2e-3

    def _inside(self, x):
        return np.all(np.abs(x) <= self.pitch / 2.0 + 1e-12, axis=1)

    def sample(self, rng, n):
        """拒绝采样：盒内均匀提议，按 Π(1+a cos u)/(1+a)^d 接受。

        与 log_prob 严格同源（FlowForge 铁律：采样器必须采自带解析密度的分布）。
        """
        lo, hi = -self.pitch / 2.0, self.pitch / 2.0
        pieces: list[np.ndarray] = []
        need = int(n)
        while need > 0:
            m = max(int(need * 1.7) + 16, 16)
            x = rng.uniform(lo, hi, size=(m, self.dim))
            u = 2.0 * np.pi * x / self.pitch
            acc = np.prod(1.0 + self.amp * np.cos(u), axis=1) / (1.0 + self.amp) ** self.dim
            keep = x[rng.random(m) < acc]
            pieces.append(keep)
            need -= keep.shape[0]
        return np.concatenate(pieces)[:n]

    def log_prob(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        inside = self._inside(x)
        u = 2.0 * np.pi * x / self.pitch
        logp = np.log1p(self.amp * np.cos(u)).sum(axis=1) - self.dim * np.log(self.pitch)
        return np.where(inside, logp, -np.inf)


def make_density(name: str) -> BaseDensity:
    """密度工厂。"""
    table = {
        "gmm8": lambda: GMM(
            "gmm8",
            means=[[2.0 * np.cos(k * np.pi / 4), 2.0 * np.sin(k * np.pi / 4)] for k in range(8)],
            stds=[[0.22, 0.22]] * 8,
            bounds=(-3.6, 3.6),
        ),
        "ring": lambda: Ring(),
        "wave": lambda: Wave(),
        "checker": lambda: SinusoidalGrid(),
        "gauss2gauss": lambda: GaussianTransfer(),
    }
    if name not in table:
        raise DensitySpecError(f"E101: unknown density {name!r}; available: {sorted(table)}")
    return table[name]()


ALL_DENSITIES = ("gmm8", "ring", "wave", "checker")
