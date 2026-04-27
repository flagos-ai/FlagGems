import importlib.util
from importlib import metadata

import torch

from . import attri_util as consts


def SkipVersion(module_name, pattern):
    if importlib.util.find_spec(module_name) is None:
        return True

    op = pattern[0]
    assert op in ("=", "<", ">"), f"Invalid comparison operator: {op}"
    try:
        M, N = pattern[1:].split(".")
        M, N = int(M), int(N)
    except Exception:
        raise ValueError("Cannot parse version number from pattern.")

    try:
        version = metadata.version(module_name)
        major, minor = map(int, version.split(".")[:2])
    except Exception:
        raise ImportError(f"Cannot determine version of module: {module_name}")

    if op == "=":
        return major == M and minor == N

    if op == "<":
        return (major, minor) < (M, N)

    return (major, minor) > (M, N)


def generate_tensor_input(shape, dtype, device):
    if dtype in consts.FLOAT_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)

    if dtype in consts.INT_DTYPES:
        return torch.randint(
            torch.iinfo(dtype).min,
            torch.iinfo(dtype).max,
            shape,
            dtype=dtype,
            device="cpu",
        ).to(device)

    if dtype in consts.BOOL_DTYPES:
        return torch.randint(0, 2, size=shape, dtype=dtype, device="cpu").to(device)

    if dtype in consts.COMPLEX_DTYPES:
        return torch.randn(shape, dtype=dtype, device=device)


def binary_input_fn(shape, dtype, device):
    inp1 = generate_tensor_input(shape, dtype, device)
    inp2 = generate_tensor_input(shape, dtype, device)
    yield inp1, inp2


def unary_input_fn(shape, dtype, device):
    yield generate_tensor_input(shape, dtype, device),
