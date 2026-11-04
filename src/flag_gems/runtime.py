# src/flag_gems/runtime.py
import torch
from flag_gems_binding import BackendType, get_default_backend


def get_compiler_backend():
    """
    运行时判断后端：
    1. 设备是Iluvatar + flagtree可用 → flagtree
    2. 其余 → aten fallback
    """
    # 环境变量强制覆盖
    import os

    force_backend = os.getenv("FLAGGEMS_BACKEND", None)
    if force_backend == "flagtree":
        return "flagtree"
    if force_backend == "aten":
        return "aten"

    # C++ binding读取默认配置
    bk = get_default_backend()
    if bk == BackendType.kFlagTree:
        return "flagtree"
    return "aten"
