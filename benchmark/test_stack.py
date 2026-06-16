import math
from typing import Generator

import pytest
import torch

from . import base, consts, utils

# Maximum number of elements per input tensor for the stack benchmark.
# Stack creates 3 inputs + 1 output = 4 tensors of the same shape, so
# peak memory is 4 × numel × dtype_bytes.  On a 32 GiB GPU with float32
# (4 bytes) this gives 4 × N × 4 ≤ ~24 GiB → N ≤ ~6 × 10^8.
# Using 2^28 (~268 M) as a conservative threshold to leave headroom for
# the Triton JIT cache, L2-flush buffer and other overhead.
_MAX_NUMEL = 2**28


class StackBenchmark(base.Benchmark):
    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Filter out shapes whose numel exceeds the safe limit for stack
        # (3 inputs + 1 output simultaneously allocated).
        self.shapes = [s for s in self.shapes if math.prod(s) <= _MAX_NUMEL]

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)

    def set_more_shapes(self):
        more_shapes_2d = [(1024, 2**i) for i in range(1, 11, 4)]
        more_shapes_3d = [(64, 64, 2**i) for i in range(0, 8, 4)]
        return more_shapes_2d + more_shapes_3d


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = utils.generate_tensor_input(shape, dtype, device)
    inp3 = utils.generate_tensor_input(shape, dtype, device)
    yield [inp1, inp2, inp3], {"dim": 0},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield [inp1, inp2, inp3], {"dim": -1},


@pytest.mark.stack
def test_stack():
    bench = StackBenchmark(
        op_name="stack",
        input_fn=_input_fn,
        torch_op=torch.stack,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
