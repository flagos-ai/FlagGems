import pytest
import torch

import flag_gems

from . import base, consts


class _NansumBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
            mask = torch.rand(shape, device=self.device) > 0.7
            x[mask] = float("nan")

            ndim = x.ndim

            yield (x,)

            if ndim >= 2:
                yield x, -1
                yield x, 0

            if ndim >= 3:
                yield x, 1


@pytest.mark.nansum
def test_benchmark_nansum():
    bench = _NansumBenchmark(
        op_name="nansum",
        torch_op=torch.nansum,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nansum)
    bench.run()
