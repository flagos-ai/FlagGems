import pytest
import torch

import flag_gems

from . import base, consts


class _NanmeanBenchmark(base.UnaryReductionBenchmark):
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


@pytest.mark.nanmean
def test_benchmark_nanmean():
    bench = _NanmeanBenchmark(
        op_name="nanmean",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean)
    bench.run()
