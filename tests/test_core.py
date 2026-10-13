"""core：种子/线程钉、错误目录、配置覆盖与校验。"""

import numpy as np
import pytest

from rectforge.core import config as cfg_mod
from rectforge.core.errors import ERROR_CATALOG, ConfigError, RectForgeError, error_class
from rectforge.core.seed import set_all
from rectforge.core.types import BenchmarkConfig


# ---- 错误目录 --------------------------------------------------------
def test_error_catalog_unique_codes():
    codes = list(ERROR_CATALOG)
    assert len(codes) == len(set(codes))
    names = [c.__name__ for c in ERROR_CATALOG.values()]
    assert len(names) == len(set(names))


def test_error_codes_monotone_catalog():
    assert ERROR_CATALOG["E000"] is RectForgeError
    assert issubclass(error_class("E500"), RectForgeError)


def test_error_class_unknown_raises():
    with pytest.raises(KeyError):
        error_class("E999")


def test_duplicate_code_registration_rejected():
    from rectforge.core import errors as err

    class _Dup(err.RectForgeError):
        code = "E100"

    with pytest.raises(RuntimeError, match="duplicate"):
        err._register(_Dup)


# ---- 种子 ------------------------------------------------------------
def test_set_all_reproducible():
    a = set_all(42)
    b = set_all(42)
    assert np.array_equal(a.normal(size=16), b.normal(size=16))


# ---- ENV 覆盖：类型映射 ----------------------------------------------
def test_env_overrides_types(monkeypatch):
    monkeypatch.setenv("RECTFORGE_STEPS", "77")
    monkeypatch.setenv("RECTFORGE_LR", "0.005")
    monkeypatch.setenv("RECTFORGE_SEEDS", "(3, 9)")
    monkeypatch.setenv("RECTFORGE_DGPS", "('ring',)")
    monkeypatch.setenv("RECTFORGE_DEMO_FAST", "1")
    cfg = cfg_mod.apply_env_overrides(BenchmarkConfig())
    assert cfg.train.steps == 77 and isinstance(cfg.train.steps, int)
    assert cfg.train.lr == 0.005 and isinstance(cfg.train.lr, float)
    assert cfg.seeds == (3, 9)
    assert cfg.dgps == ("ring",)
    assert cfg.demo_fast is True


def test_env_override_unknown_train_field_ignored(monkeypatch):
    monkeypatch.setenv("RECTFORGE_NOT_A_FIELD", "1")
    cfg = cfg_mod.apply_env_overrides(BenchmarkConfig())
    assert cfg.train.steps == 1500  # 默认不变


def test_coerce_unknown_field_raises():
    with pytest.raises(ConfigError, match="E103"):
        cfg_mod._coerce("no_such_field", "1")


# ---- schema 校验 -----------------------------------------------------
def test_validate_bad_path(monkeypatch):
    monkeypatch.setenv("RECTFORGE_PATH", "cubic")
    with pytest.raises(ConfigError, match="E104"):
        cfg_mod.load_config()


def test_validate_bad_coupling(monkeypatch):
    monkeypatch.setenv("RECTFORGE_COUPLING", "otimal")
    with pytest.raises(ConfigError, match="E105"):
        cfg_mod.load_config()


def test_validate_nonpositive_steps(monkeypatch):
    monkeypatch.setenv("RECTFORGE_STEPS", "0")
    with pytest.raises(ConfigError, match="E107"):
        cfg_mod.load_config()


def test_validate_empty_dgps(monkeypatch):
    monkeypatch.setenv("RECTFORGE_DGPS", "()")
    with pytest.raises(ConfigError, match="E110"):
        cfg_mod.load_config()


def test_validate_bad_sampler(monkeypatch):
    monkeypatch.setenv("RECTFORGE_FLAGSHIP_SAMPLER", "dopri5")
    with pytest.raises(ConfigError, match="E109"):
        cfg_mod.load_config()


def test_config_error_hierarchy():
    assert issubclass(ConfigError, RectForgeError)
    assert ConfigError.code == "E103"


def test_env_prefix_only(monkeypatch):
    monkeypatch.setenv("RECTFORGE_STEPS", "55")
    monkeypatch.setenv("RECTFORGEXYZ_STEPS", "999")
    cfg = cfg_mod.apply_env_overrides(BenchmarkConfig())
    assert cfg.train.steps == 55
