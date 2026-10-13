"""RectForge - flow matching / rectified flow / stochastic interpolants.

纯 NumPy、确定性、CPU-only。import 本包即钉 BLAS 线程（尽力而为；
严格确定性请在任何 numpy 导入前调用 core.seed.pin_threads()）。
"""
from __future__ import annotations

from .core.seed import pin_threads as _pin

_pin()

__version__ = "0.1.0"
