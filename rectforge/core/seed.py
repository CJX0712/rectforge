"""全局确定性入口：BLAS 线程钉 1 + 显式 Generator 约定。

铁律：进程内第一次 import numpy 之前必须完成 pin_threads()，
否则 OpenBLAS 多线程求和顺序不确定（MatrixForge 实测非确定性 1.5e-14）。
cli.py / examples/run_demo.py / tests/conftest.py 均在文件最顶部调用。
"""
from __future__ import annotations

import os
import random

_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def pin_threads() -> None:
    """把 BLAS/OpenMP 线程数钉为 1（须在 numpy 首次导入前调用）。"""
    for key in _THREAD_ENV:
        os.environ[key] = "1"


def set_all(seed: int):
    """唯一 seed 入口：钉线程 + 置 random 模块种子，返回显式 Generator。

    约定：仓库内所有随机性必须来自显式传入的 Generator，
    禁止使用 np.random 全局状态（保证跨模块、跨进程一致）。
    """
    pin_threads()
    import numpy as np

    random.seed(seed)
    return np.random.default_rng(seed)
