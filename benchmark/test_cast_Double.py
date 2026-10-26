import pytest
import torch

import flag_gems

from .base import Benchmark


class CastDoubleBenchmark(Benchmark):
    def get_input_iter(self, cur_dtype):
        for shape in [(1024,), (65536,), (1048576,), (16777216,)]:
            yield (torch.randn(shape, device=self.device, dtype=cur_dtype),)
        x = torch.randn((1024, 1024), device=self.device, dtype=cur_dtype)
        yield (x.t(),)
        yield (x[:, ::2],)


@pytest.mark.cast_Double
def test_cast_Double():
    bench = CastDoubleBenchmark(
        op_name="cast_Double",
        torch_op=torch.ops.aten._cast_Double.default,
        gems_op=flag_gems.cast_Double,
        dtypes=[torch.float16, torch.bfloat16, torch.float32],
    )
    bench.run()
