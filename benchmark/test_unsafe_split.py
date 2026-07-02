from typing import Generator

import pytest
import torch

import flag_gems
from flag_gems.config import c_operators, has_c_extension
from flag_gems.ops.unsafe_split import unsafe_split, unsafe_split_with_sizes

from . import base

# View-only split is dtype independent; keep the CI benchmark to Ascend-friendly
# low precision dtypes that were requested for performance tracking.
BENCH_DTYPES = [torch.float16, torch.bfloat16]  # Ascend perf path covers fp16/bf16.
BENCH_SHAPES = [
    (1024,),
    (64, 64),
    (128, 256),
    (16, 128, 64),
]


def _torch_device_available(device_name):
    if device_name == "cpu":
        return False
    if device_name == "npu" and not hasattr(torch, "npu"):
        try:
            __import__("torch_npu")
        except ImportError:
            return False
    device_mod = getattr(torch, device_name, None)
    if device_mod is None or not hasattr(device_mod, "is_available"):
        return False
    try:
        return bool(device_mod.is_available())
    except Exception:
        return False


def _test_device():
    if _torch_device_available(flag_gems.device):
        return torch.device(flag_gems.device)
    for device_name in ("npu", "cuda"):
        if _torch_device_available(device_name):
            return torch.device(device_name)
    return None


TEST_DEVICE = _test_device()


pytestmark = pytest.mark.skipif(
    not (
        has_c_extension
        and c_operators is not None
        and hasattr(c_operators, "unsafe_split")
        and hasattr(c_operators, "unsafe_split_with_sizes")
        and TEST_DEVICE is not None
    ),
    reason="unsafe_split Python wrapper requires an available FlagGems C extension backend device",
)


def _make_input(shape, dtype, device):
    return torch.zeros(shape, dtype=dtype, device=device)


def _split_input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    dim = -1 if len(shape) > 1 else 0
    split_size = max(1, inp.size(dim) // 4)
    yield inp, split_size, dim


def _split_with_sizes_input_fn(shape, dtype, device):
    inp = _make_input(shape, dtype, device)
    dim = -1 if len(shape) > 1 else 0
    dim_size = inp.size(dim)
    first = dim_size // 4
    second = dim_size // 2
    split_sizes = [first, 0, second, dim_size - first - second]
    yield inp, split_sizes, dim


class UnsafeSplitBenchmark(base.GenericBenchmark):
    device = str(TEST_DEVICE) if TEST_DEVICE is not None else base.device
    DEFAULT_SHAPES = BENCH_SHAPES
    DEFAULT_SHAPE_DESC = "(B), M, N"

    def init_default_config(self):
        self.shapes = self.DEFAULT_SHAPES

    def init_user_config(self):
        self.mode = base.Config.mode
        self.set_dtypes(base.Config.user_desired_dtypes)
        self.set_metrics(base.Config.user_desired_metrics)
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            if len(shape) == 0:
                continue
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.unsafe_split
def test_unsafe_split():
    bench = UnsafeSplitBenchmark(
        op_name="unsafe_split",
        torch_op=torch.ops.aten.unsafe_split.Tensor,
        gems_op=unsafe_split,
        input_fn=_split_input_fn,
        dtypes=BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.unsafe_split
def test_unsafe_split_with_sizes():
    bench = UnsafeSplitBenchmark(
        op_name="unsafe_split",
        torch_op=torch.ops.aten.unsafe_split_with_sizes.default,
        gems_op=unsafe_split_with_sizes,
        input_fn=_split_with_sizes_input_fn,
        dtypes=BENCH_DTYPES,
    )
    bench.run()
