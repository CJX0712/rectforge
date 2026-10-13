"""错误目录 E100~E599。

ValuaForge 教训：同名错误类会静默遮蔽先定义者（重码之外，重名更隐蔽）。
因此注册时同时检测「重码」与「重名」，撞名即 RuntimeError 立即炸。
"""
from __future__ import annotations

ERROR_CATALOG: dict[str, type] = {}


class RectForgeError(Exception):
    """所有 RectForge 错误的基类。"""

    code = "E000"


class ConfigError(RectForgeError):
    """配置覆盖/校验错误（ENV_RECTFORGE_* 解析、schema 断言）。"""

    code = "E103"


class DataError(RectForgeError):
    """合成密度/数据管线错误。"""

    code = "E100"


class DensitySpecError(DataError):
    """密度规格非法。"""

    code = "E101"


class NormalizationError(DataError):
    """密度归一化自检失败（积分 != 1）。"""

    code = "E102"


class PathError(RectForgeError):
    """概率路径/插值算子错误。"""

    code = "E200"


class CouplingError(RectForgeError):
    """耦合构造错误。"""

    code = "E201"


class ModelError(RectForgeError):
    """网络/参数错误。"""

    code = "E300"


class GradientError(ModelError):
    """梯度校验失败。"""

    code = "E301"


class TrainingError(RectForgeError):
    """训练过程错误。"""

    code = "E400"


class SamplingError(RectForgeError):
    """采样/积分错误。"""

    code = "E500"


class MetricError(RectForgeError):
    """评测指标错误。"""

    code = "E501"


def _register(cls: type) -> type:
    seen_codes = {c.code: c.__name__ for c in ERROR_CATALOG.values()}
    if cls.code in seen_codes:
        raise RuntimeError(
            f"duplicate error code {cls.code}: {cls.__name__} vs {seen_codes[cls.code]}"
        )
    if cls.__name__ in ERROR_CATALOG.values():
        raise RuntimeError(f"duplicate error class name {cls.__name__}")
    ERROR_CATALOG[cls.code] = cls
    return cls


for _cls in (
    RectForgeError,
    ConfigError,
    DataError,
    DensitySpecError,
    NormalizationError,
    PathError,
    CouplingError,
    ModelError,
    GradientError,
    TrainingError,
    SamplingError,
    MetricError,
):
    _register(_cls)


def error_class(code: str) -> type:
    if code not in ERROR_CATALOG:
        raise KeyError(f"unknown error code {code}")
    return ERROR_CATALOG[code]
