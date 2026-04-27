from typing import Generator

import pytest
import torch

import flag_gems

from . import performance_utils as base

fp64_is_supported = flag_gems.runtime.device.support_fp64


class ToCopyBenchmark(base.UnaryPointwiseBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp = torch.randn(shape, dtype=torch.float32, device=self.device)
            yield inp, {"dtype": dtype}


@pytest.mark.to_copy
def test_to_copy():
    if fp64_is_supported:
        dtypes = [torch.float16, torch.bfloat16, torch.float64]
    else:
        dtypes = [torch.float16, torch.bfloat16]

    bench = ToCopyBenchmark(
        op_name="to_copy",
        torch_op=torch.ops.aten._to_copy,
        dtypes=dtypes,
    )
    bench.run()
