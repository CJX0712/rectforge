"""RectForge 核心层。"""
from __future__ import annotations

from . import errors as _errors  # noqa: F401  (注册错误目录)
from .seed import pin_threads, set_all

__all__ = ["pin_threads", "set_all"]
