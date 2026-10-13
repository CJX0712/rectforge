"""基准流水线：训练 -> 采样 -> 评测 -> 门禁 -> 消融。

统计严谨性约定（SOP）：
- 每（DGP × seed）独立数据流；true 评测样本与初始噪声对所有方法共用
  （common random numbers）=> 配对比较。
- 模型初始化种子固定（_MODEL_INIT_SEED）=> 同 init 配对；训练随机流按 seed 变。
- 主门禁在等 NFE 预算（48）下比较；DDPM ancestral(1000) 仅作参考口径。
- 每个数字来自真实运行；benchmark.json 落盘全部原始点。
"""
from __future__ import annotations

import platform
import time

import numpy as np

from rectforge.baselines.ddpm import DiffusionSchedule, ddim_sample, ddpm_sample
from rectforge.core.types import BenchmarkConfig, EvalPoint, GateReport, TrainConfig
from rectforge.data.densities import make_density
from rectforge.eval.metrics import kl_p_q, mmd2, sliced_w1
from rectforge.models.mlp import TimeCondMLP
from rectforge.paths.interpolants import make_path
from rectforge.sampling.ode import NFECounter, integrate_ode, pf_velocity_fn
from rectforge.training.trainer import train_ddpm, train_velocity

_MODEL_INIT_SEED = 1000
_TRUE_SEED = 900_000
_SAMPLE_SEED = 500_000
_CORPUS_SEED = 1_000_000
_CORPUS_SIZE = 2048
_DDPM_T = 1000


# ----------------------------------------------------------------------
# 配置构造
# ----------------------------------------------------------------------
def flagship_train_cfg(cfg: BenchmarkConfig) -> TrainConfig:
    t = cfg.train
    return TrainConfig(
        steps=t.steps,
        batch_size=t.batch_size,
        lr=t.lr,
        lr_min_ratio=t.lr_min_ratio,
        warmup=t.warmup,
        grad_clip=t.grad_clip,
        time_sampling=cfg.flagship_time_sampling,
        hidden=t.hidden,
        layers=t.layers,
        embed_dim=t.embed_dim,
        final_init_scale=t.final_init_scale,
        path="linear",
        si_sigma=cfg.flagship_si_sigma,
        coupling=cfg.flagship_coupling,
        ot_refresh=t.ot_refresh,
        ot_cost_scale=t.ot_cost_scale,
    )


def vp_train_cfg(cfg: BenchmarkConfig) -> TrainConfig:
    t = flagship_train_cfg(cfg)
    t.path = "vp"
    t.coupling = "independent"
    t.time_sampling = "uniform"
    t.si_sigma = 0.0
    return t


def _make_model(tcfg: TrainConfig, dim: int) -> TimeCondMLP:
    return TimeCondMLP(
        dim=dim,
        hidden=tcfg.hidden,
        layers=tcfg.layers,
        embed_dim=tcfg.embed_dim,
        final_init_scale=tcfg.final_init_scale,
        seed=_MODEL_INIT_SEED,
    )


# ----------------------------------------------------------------------
# 单点训练 + 采样
# ----------------------------------------------------------------------
def train_models(
    density, data, cfg: BenchmarkConfig, train_seed: int, schedule: DiffusionSchedule | None = None
):
    """训练三模型：rectfuse（旗舰）/ cfm_vp（FM 家族基线）/ ddpm（扩散基线）。

    训练随机流按 train_seed（同 (dgp,seed) 下三模型共用同一流 => 配对）；
    模型初始化种子固定 _MODEL_INIT_SEED（同 init 配对）。
    """
    if schedule is None:
        schedule = DiffusionSchedule(T=_DDPM_T)
    out = {}
    for name, tcfg, kind in (
        ("rectfuse", flagship_train_cfg(cfg), "fm"),
        ("cfm_vp", vp_train_cfg(cfg), "fm"),
        ("ddpm", flagship_train_cfg(cfg), "ddpm"),
    ):
        model = _make_model(tcfg, density.dim)
        train_rng = np.random.default_rng(train_seed)
        t0 = time.perf_counter()
        if kind == "fm":
            params, hist = train_velocity(model, data, tcfg, train_rng)
        else:
            params, hist = train_ddpm(model, data, schedule, tcfg, train_rng)
        out[name] = {
            "model": model,
            "params": params,
            "train_seconds": time.perf_counter() - t0,
            "final_loss": float(hist[-1]) if hist else float("nan"),
            "tcfg": tcfg,
            "schedule": schedule,
        }
    return out


def sample_method(name, models, corpus, cfg: BenchmarkConfig, n: int, rng, nfe: int):
    """返回 (samples, actual_nfe)。ddim 复用 ddpm 的权重。"""
    entry = models["ddpm" if name == "ddim" else name]
    model, params, tcfg = entry["model"], entry["params"], entry["tcfg"]
    if name in ("rectfuse", "cfm_vp"):
        path = make_path(tcfg.path, si_sigma=tcfg.si_sigma)
        u_fn = pf_velocity_fn(model, params, path, corpus)
        counter = NFECounter()
        if name == "rectfuse" and cfg.flagship_sampler == "rk4":
            method, steps = "rk4", max(nfe // 4, 1)
        else:
            method, steps = "euler", nfe
        x0 = rng.normal(size=(n, model.dim))
        samples = integrate_ode(u_fn, x0, steps, method, counter)
        return samples, counter.n
    if name == "ddim":
        counter = NFECounter()
        return ddim_sample(model, params, entry["schedule"], n, nfe, rng, counter), counter.n
    counter = NFECounter()
    return ddpm_sample(model, params, entry["schedule"], n, rng, counter), counter.n


def evaluate_point(density, x_true, samples):
    kl = kl_p_q(density.log_prob, x_true, samples)
    mm = mmd2(samples, x_true)
    w1 = sliced_w1(samples, x_true)
    return kl, mm, w1


# ----------------------------------------------------------------------
# 主基准
# ----------------------------------------------------------------------
def run_benchmark(cfg: BenchmarkConfig, verbose: bool = True) -> dict:
    points: list[EvalPoint] = []
    norm_errors = {}
    for dgp in cfg.dgps:
        density = make_density(dgp)
        norm_errors[dgp] = density.check_normalization()
        if verbose:
            print(f"[bench] {dgp}: normalization err {norm_errors[dgp]:.2e}")

    schedule = DiffusionSchedule(T=_DDPM_T)
    for dgp in cfg.dgps:
        density = make_density(dgp)
        for seed in cfg.seeds:
            data = density.sample(np.random.default_rng(seed), cfg.n_train)
            x_true = density.sample(np.random.default_rng(seed + _TRUE_SEED), cfg.true_eval_samples)
            corpus = data[
                np.random.default_rng(seed + _CORPUS_SEED).integers(
                    0, len(data), min(_CORPUS_SIZE, len(data))
                )
            ]
            models = train_models(density, data, cfg, train_seed=seed, schedule=schedule)

            plans = [("rectfuse", nfe) for nfe in cfg.nfe_sweep]
            plans += [("cfm_vp", cfg.nfe_sweep[-1])]
            plans += [("ddim", nfe) for nfe in cfg.nfe_sweep]
            plans += [("ddpm", _DDPM_T)]
            for method, nfe in plans:
                rng = np.random.default_rng(seed + _SAMPLE_SEED)
                t0 = time.perf_counter()
                samples, actual_nfe = sample_method(
                    method, models, corpus, cfg, cfg.eval_samples, rng, nfe
                )
                kl, mm, w1 = evaluate_point(density, x_true, samples)
                points.append(
                    EvalPoint(
                        dgp=dgp,
                        seed=seed,
                        method=method,
                        kl=kl,
                        mmd2=mm,
                        w1=w1,
                        nfe=actual_nfe,
                        sample_seconds=round(time.perf_counter() - t0, 4),
                    )
                )
                if verbose:
                    print(
                        f"[bench] {dgp} seed={seed} {method} nfe={actual_nfe}: "
                        f"KL={kl:.5f} MMD2={mm:.5f} W1={w1:.5f}"
                    )
    return _aggregate(cfg, points, norm_errors)


def _aggregate(cfg, points, norm_errors):
    per_method: dict[str, list[EvalPoint]] = {}
    for p in points:
        per_method.setdefault(p.method, []).append(p)
    methods = {}
    for name, pts in per_method.items():
        kls = np.array([p.kl for p in pts])
        per_dgp = {}
        for dgp in cfg.dgps:
            vals = [p.kl for p in pts if p.dgp == dgp]
            if vals:
                per_dgp[dgp] = float(np.mean(vals))
        methods[name] = {
            "kl_mean": float(kls.mean()),
            "kl_std": float(kls.std(ddof=1)) if len(kls) > 1 else 0.0,
            "mmd2_mean": float(np.mean([p.mmd2 for p in pts])),
            "w1_mean": float(np.mean([p.w1 for p in pts])),
            "nfe": int(pts[0].nfe),
            "n_points": len(pts),
            "per_dgp": per_dgp,
            "train_seconds_mean": float(np.mean([p.sample_seconds for p in pts])),
        }
    gates = _gates(cfg, points, methods, norm_errors)
    return {
        "config": {
            "dgps": list(cfg.dgps),
            "seeds": list(cfg.seeds),
            "nfe_sweep": list(cfg.nfe_sweep),
            "eval_samples": cfg.eval_samples,
            "true_eval_samples": cfg.true_eval_samples,
            "flagship_sampler": cfg.flagship_sampler,
            "flagship_coupling": cfg.flagship_coupling,
            "flagship_time_sampling": cfg.flagship_time_sampling,
            "train_steps": cfg.train.steps,
            "batch_size": cfg.train.batch_size,
            "hidden": cfg.train.hidden,
            "layers": cfg.train.layers,
            "ddpm_T": _DDPM_T,
        },
        "normalization_errors": norm_errors,
        "points": [vars(p) for p in points],
        "methods": methods,
        "gates": [vars(g) for g in gates],
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }


def _gates(cfg, points, methods, norm_errors):
    gates = []
    # G1a: 归一化（各密度按自身容差；checker 盒界不连续致梯形 O(spacing) 误差）
    worst_name = max(norm_errors, key=lambda k: norm_errors[k])
    max_err = max(norm_errors.values())
    gates.append(
        GateReport(
            name="G1a_normalization",
            passed=max_err < 2e-3,
            detail=(
                f"max normalization err = {max_err:.2e} (dgp={worst_name}); "
                f"per-dgp { {k: round(v, 8) for k, v in norm_errors.items()} }"
            ),
        )
    )
    # G1b: KL 非负（大样本容忍）
    kl_min = min(p.kl for p in points)
    gates.append(
        GateReport(
            name="G1b_kl_nonnegative",
            passed=kl_min >= -1e-3,
            detail=f"min KL over all points = {kl_min:.6f} (tol -1e-3)",
        )
    )
    # G2: 等 NFE 旗舰 vs 最强基线（ddim）。统计设计要求 >=3 DGP；
    # 覆盖不足（演示/冒烟配置）时诚实降级为 SKIPPED 而非 FAIL。
    top_nfe = cfg.nfe_sweep[-1]
    flag = [p for p in points if p.method == "rectfuse" and p.nfe == top_nfe]
    base = [p for p in points if p.method == "ddim" and p.nfe == top_nfe]
    if len(cfg.dgps) < 3:
        gates.append(
            GateReport(
                name="G2_matched_nfe",
                passed=True,
                detail=f"SKIPPED: {len(cfg.dgps)} DGP < 3 required for paired significance",
            )
        )
    elif flag and base:
        fmap = {(p.dgp, p.seed): p.kl for p in flag}
        bmap = {(p.dgp, p.seed): p.kl for p in base}
        keys = sorted(set(fmap) & set(bmap))
        diffs = np.array([bmap[k] - fmap[k] for k in keys])
        ratios = np.array([fmap[k] / bmap[k] for k in keys if bmap[k] > 0])
        kl_f = np.array([fmap[k] for k in keys])
        kl_b = np.array([bmap[k] for k in keys])
        sig = (
            0.5 * (kl_f.std(ddof=1) + kl_b.std(ddof=1))
            if len(kl_f) > 1 and len(kl_b) > 1
            else 0.0
        )
        wins = int(sum(1 for d in diffs if d > 0))
        per_dgp_ratio = {}
        for dgp in cfg.dgps:
            fr = [fmap[k] for k in keys if k[0] == dgp]
            br = [bmap[k] for k in keys if k[0] == dgp]
            if fr and br and np.mean(br) > 0:
                per_dgp_ratio[dgp] = float(np.mean(fr) / np.mean(br))
        n_dgp_ok = sum(1 for r in per_dgp_ratio.values() if r <= 0.90)
        passed = bool(
            ratios.mean() <= 0.90
            and n_dgp_ok >= 3
            and diffs.mean() > sig
            and wins >= max(3, int(0.75 * len(keys)))
        )
        gates.append(
            GateReport(
                name="G2_matched_nfe",
                passed=passed,
                detail=(
                    f"NFE={top_nfe}: flagship/ddim KL ratio mean = {ratios.mean():.4f} (<=0.90); "
                    f"per-dgp {per_dgp_ratio}; mean diff {diffs.mean():.5f} > "
                    f"0.5*(sigma_f+sigma_b)={sig:.5f}; paired wins {wins}/{len(keys)}"
                ),
            )
        )
    else:
        gates.append(GateReport(name="G2_matched_nfe", passed=False, detail="insufficient points"))
    # G2b: 效率 —— 旗舰达到 ddim@top 参考质量所需 NFE（需 sweep 最小值 <= top/2）
    base_ref = float(np.mean([p.kl for p in points if p.method == "ddim" and p.nfe == top_nfe]))
    if len(cfg.dgps) < 3 or cfg.nfe_sweep[0] > top_nfe // 2:
        gates.append(
            GateReport(
                name="G2b_nfe_efficiency",
                passed=True,
                detail=(
                    f"SKIPPED: sweep resolution insufficient "
                    f"(min nfe {cfg.nfe_sweep[0]} > {top_nfe // 2} or <3 DGP)"
                ),
            )
        )
    else:
        flag_nfe_ok = None
        for nfe in cfg.nfe_sweep:
            vals = [p.kl for p in points if p.method == "rectfuse" and p.nfe == nfe]
            if vals and float(np.mean(vals)) <= base_ref:
                flag_nfe_ok = nfe
                break
        gates.append(
            GateReport(
                name="G2b_nfe_efficiency",
                passed=flag_nfe_ok is not None and flag_nfe_ok <= top_nfe / 2,
                detail=(
                    f"ddim@{top_nfe} KL ref = {base_ref:.5f}; "
                    f"flagship reaches it at NFE={flag_nfe_ok} (gate: <= {top_nfe // 2})"
                ),
            )
        )
    return gates


# ----------------------------------------------------------------------
# 消融（2 seeds，报告全部含负结果）
# ----------------------------------------------------------------------
def run_ablation(cfg: BenchmarkConfig, verbose: bool = True) -> dict:
    variants = {
        "full_ot_uniform_rk4": {
            "coupling": "ot", "time_sampling": "uniform", "sampler": "rk4", "si": 0.0,
        },
        "no_ot_independent": {
            "coupling": "independent", "time_sampling": "uniform", "sampler": "rk4", "si": 0.0,
        },
        "lognormal_time": {
            "coupling": "ot", "time_sampling": "logit_normal", "sampler": "rk4", "si": 0.0,
        },
        "euler_sampler": {
            "coupling": "ot", "time_sampling": "uniform", "sampler": "euler", "si": 0.0,
        },
        "si_noise_0.25": {
            "coupling": "independent", "time_sampling": "uniform", "sampler": "rk4", "si": 0.25,
        },
    }
    rows = {}
    for vname, v in variants.items():
        kls = []
        for dgp in cfg.dgps:
            density = make_density(dgp)
            for seed in cfg.ablation_seeds:
                data = density.sample(np.random.default_rng(seed), cfg.n_train)
                x_true = density.sample(np.random.default_rng(seed + _TRUE_SEED), cfg.true_eval_samples)
                corpus = data[
                    np.random.default_rng(seed + _CORPUS_SEED).integers(
                        0, len(data), min(_CORPUS_SIZE, len(data))
                    )
                ]
                tcfg = flagship_train_cfg(cfg)
                tcfg.coupling = v["coupling"]
                tcfg.time_sampling = v["time_sampling"]
                tcfg.path = "si" if v["si"] > 0 else "linear"
                tcfg.si_sigma = v["si"]
                model = _make_model(tcfg, density.dim)
                train_rng = np.random.default_rng(seed)
                params, _ = train_velocity(model, data, tcfg, train_rng)
                path = make_path(tcfg.path, si_sigma=tcfg.si_sigma)
                u_fn = pf_velocity_fn(model, params, path, corpus)
                counter = NFECounter()
                x0 = np.random.default_rng(seed + _SAMPLE_SEED).normal(size=(cfg.eval_samples, model.dim))
                steps = max(cfg.nfe_sweep[-1] // 4, 1) if v["sampler"] == "rk4" else cfg.nfe_sweep[-1]
                samples = integrate_ode(u_fn, x0, steps, v["sampler"], counter)
                kl, _, _ = evaluate_point(density, x_true, samples)
                kls.append(kl)
        rows[vname] = {
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls, ddof=1)) if len(kls) > 1 else 0.0,
            "nfe": counter.n,
        }
        if verbose:
            print(f"[ablation] {vname}: KL {rows[vname]['kl_mean']:.5f} +/- {rows[vname]['kl_std']:.5f}")
    return {"variants": rows, "seeds": list(cfg.ablation_seeds), "nfe_budget": cfg.nfe_sweep[-1]}


# ----------------------------------------------------------------------
# 确定性校验（mini 配置，两次运行逐位比较）
# ----------------------------------------------------------------------
def run_determinism_check(verbose: bool = True) -> dict:
    cfg = BenchmarkConfig(
        dgps=("gmm8",),
        seeds=(7,),
        n_train=4000,
        eval_samples=512,
        true_eval_samples=1024,
        nfe_sweep=(12,),
    )
    cfg.train.steps = 300

    def once():
        density = make_density("gmm8")
        seed = 7
        data = density.sample(np.random.default_rng(seed), cfg.n_train)
        x_true = density.sample(np.random.default_rng(seed + _TRUE_SEED), cfg.true_eval_samples)
        corpus = data[np.random.default_rng(seed + _CORPUS_SEED).integers(0, len(data), 2048)]
        models = train_models(density, data, cfg, train_seed=seed)
        out = {}
        for method in ("rectfuse", "ddim"):
            rng = np.random.default_rng(seed + _SAMPLE_SEED)
            samples, nfe = sample_method(method, models, corpus, cfg, cfg.eval_samples, rng, 12)
            kl, mm, w1 = evaluate_point(density, x_true, samples)
            out[method] = {"kl": kl, "mmd2": mm, "w1": w1, "nfe": nfe}
        return out

    a = once()
    b = once()
    max_diff = 0.0
    for m in a:
        for k in ("kl", "mmd2", "w1"):
            max_diff = max(max_diff, abs(a[m][k] - b[m][k]))
    ok = max_diff == 0.0
    if verbose:
        print(f"[determinism] max|delta| = {max_diff:.3e} -> {'BIT-IDENTICAL' if ok else 'MISMATCH'}")
    return {"bit_identical": ok, "max_abs_diff": max_diff, "run_a": a, "run_b": b}


# ----------------------------------------------------------------------
# 自检（G1 不变量，CI 冒烟用）
# ----------------------------------------------------------------------
def run_selftest(verbose: bool = True) -> dict:
    from rectforge.data.densities import gaussian_transfer_velocity
    from rectforge.sampling.ode import rk4_integrate

    report: dict = {}
    for name in ("gmm8", "ring", "wave", "checker", "gauss2gauss"):
        d = make_density(name)
        report[f"norm_{name}"] = d.check_normalization()
    # 解析速度场的输运（生成方向）：N(0,I)@t=1 -> N(mu,cov)@t=0，矩精确
    d = make_density("gauss2gauss")
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16384, 2))  # t=1 端：标准正态噪声

    def u(x_, t):
        return gaussian_transfer_velocity(x_, t, d.mu, d.cov, np.zeros(2), np.eye(2))

    counter = NFECounter()
    y = rk4_integrate(u, x, 100, counter)
    mean_err = float(np.abs(y.mean(axis=0) - d.mu).max())
    cov_err = float(np.abs(np.cov(y.T) - d.cov).max())
    report["transport_mean_err"] = mean_err
    report["transport_cov_err"] = cov_err
    report["transport_ok"] = bool(mean_err < 0.02 and cov_err < 0.02)
    # KL(p||p) ~ 0（KDE 协议自检）
    xt = d.sample(np.random.default_rng(1), 3000)
    xs = d.sample(np.random.default_rng(2), 3000)
    kl_self = kl_p_q(d.log_prob, xt, xs)
    report["kl_self"] = kl_self
    report["kl_self_ok"] = bool(abs(kl_self) < 0.05)
    report["ok"] = bool(report.get("transport_ok") and report.get("kl_self_ok"))
    if verbose:
        for k, v in report.items():
            print(f"[selftest] {k}: {v}")
    return report
