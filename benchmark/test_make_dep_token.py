from typing import Generator

import pytest
import torch

from . import base, consts


class MakeDepTokenBenchmark(base.Benchmark):
    """Benchmark for _make_dep_token - creates scalar dependency token.

    Note: _make_dep_token has no CUDA kernel in PyTorch, so we benchmark
    latency_base against torch.zeros(1) as the closest comparable creation op,
    and measure the Gems path separately via gems_op.
    """

    def set_more_shapes(self):
        return [(), (1,), (1, 1)]

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield {"dtype": cur_dtype, "device": self.device},


@pytest.mark.make_dep_token
def test_make_dep_token():
    bench = MakeDepTokenBenchmark(
        op_name="make_dep_token",
        torch_op=lambda dtype, device: torch.zeros(1, dtype=dtype, device=device),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
