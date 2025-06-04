import importlib

import torch
from packaging import version


def has_c_extension() -> bool:
    try:
        import flag_gems.ext_ops  # noqa: F401

        return True
    except ImportError:
        return False


def is_torch_version_ge(min_ver: str) -> bool:
    return version.parse(torch.__version__) >= version.parse(min_ver)


def has_vllm() -> bool:
    return importlib.util.find_spec("vllm") is not None
