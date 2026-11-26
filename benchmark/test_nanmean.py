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


class _NanmeanOutBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=cur_dtype, device=self.device) * 10
            x[torch.rand(shape, device=self.device) > 0.7] = float("nan")

            yield x, {"out": torch.empty((), dtype=cur_dtype, device=self.device)}

            if x.ndim >= 2:
                for dim in (-1, 0):
                    out_shape = torch.nanmean(x, dim=dim).shape
                    out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                    yield x, dim, {"out": out}

            if x.ndim >= 3:
                out_shape = torch.nanmean(x, dim=1).shape
                out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                yield x, 1, {"out": out}


@pytest.mark.nanmean
def test_benchmark_nanmean():
    bench = _NanmeanBenchmark(
        op_name="nanmean",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean)
    bench.run()


@pytest.mark.nanmean_out
def test_benchmark_nanmean_out():
    bench = _NanmeanOutBenchmark(
        op_name="nanmean_out",
        torch_op=torch.nanmean,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean_out)
    bench.run()
