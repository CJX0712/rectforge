"""配置：ENV_RECTFORGE_* 环境变量覆盖 + schema 校验。

MatrixForge 教训：PEP 563 使 dataclass 字段 f.type 变成字符串，
按字段名字符串映射做类型转换（int/float/tuple/str），并配「覆盖后类型断言」单测。
"""
from __future__ import annotations

import ast
import os
from dataclasses import fields

from .errors import ConfigError
from .types import BenchmarkConfig, TrainConfig

_ENV_PREFIX = "RECTFORGE_"

_INT_FIELDS = {
    "steps",
    "batch_size",
    "warmup",
    "hidden",
    "layers",
    "embed_dim",
    "ot_refresh",
    "n_train",
    "eval_samples",
    "true_eval_samples",
}
_FLOAT_FIELDS = {
    "lr",
    "lr_min_ratio",
    "grad_clip",
    "si_sigma",
    "final_init_scale",
    "ot_cost_scale",
    "flagship_si_sigma",
}
_TUPLE_INT_FIELDS = {"seeds", "ablation_seeds"}
_TUPLE_STR_FIELDS = {"dgps", "nfe_sweep"}
_STR_FIELDS = {
    "time_sampling",
    "path",
    "coupling",
    "flagship_sampler",
    "flagship_coupling",
    "flagship_time_sampling",
}
_BOOL_FIELDS = {"demo_fast"}

_VALID_PATH = ("linear", "vp", "si")
_VALID_COUPLING = ("independent", "ot")
_VALID_TIME = ("uniform", "logit_normal")
_VALID_SAMPLER = ("euler", "rk4")


def _coerce(name: str, raw: str):  # noqa: PLR0911 - 按字段集合分派的类型映射，分支即文档
    if name in _INT_FIELDS:
        return int(raw)
    if name in _FLOAT_FIELDS:
        return float(raw)
    if name in _TUPLE_INT_FIELDS:
        vals = ast.literal_eval(raw)
        return tuple(int(v) for v in vals)
    if name in _TUPLE_STR_FIELDS:
        vals = ast.literal_eval(raw)
        if name == "nfe_sweep":
            return tuple(int(v) for v in vals)
        return tuple(str(v) for v in vals)
    if name in _STR_FIELDS:
        return str(raw)
    if name in _BOOL_FIELDS:
        return raw.strip().lower() in ("1", "true", "yes")
    raise ConfigError(f"E103: no type mapping for config field {name!r}")


def apply_env_overrides(cfg: BenchmarkConfig) -> BenchmarkConfig:
    """把 ENV_RECTFORGE_* 覆盖写进 cfg（含 train 子配置）。"""
    train_field_names = {f.name for f in fields(TrainConfig)}
    for key, raw in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        name = key[len(_ENV_PREFIX):].lower()
        if name in train_field_names:
            value = _coerce(name, raw)
            setattr(cfg.train, name, value)
        else:
            top_names = {f.name for f in fields(BenchmarkConfig)}
            if name in top_names:
                setattr(cfg, name, _coerce(name, raw))
    _validate(cfg)
    return cfg


def _validate(cfg: BenchmarkConfig) -> None:
    t = cfg.train
    if t.path not in _VALID_PATH:
        raise ConfigError(f"E104: train.path must be one of {_VALID_PATH}, got {t.path!r}")
    if t.coupling not in _VALID_COUPLING:
        raise ConfigError(f"E105: train.coupling must be one of {_VALID_COUPLING}, got {t.coupling!r}")
    if t.time_sampling not in _VALID_TIME:
        raise ConfigError(f"E106: train.time_sampling must be one of {_VALID_TIME}")
    if t.steps <= 0 or t.batch_size <= 0:
        raise ConfigError("E107: steps/batch_size must be positive")
    if t.hidden <= 0 or t.layers <= 0:
        raise ConfigError("E108: hidden/layers must be positive")
    if cfg.flagship_sampler not in _VALID_SAMPLER:
        raise ConfigError(f"E109: flagship_sampler must be one of {_VALID_SAMPLER}")
    if not cfg.dgps:
        raise ConfigError("E110: dgps must be non-empty")
    if not cfg.seeds:
        raise ConfigError("E111: seeds must be non-empty")


def load_config() -> BenchmarkConfig:
    return apply_env_overrides(BenchmarkConfig())
