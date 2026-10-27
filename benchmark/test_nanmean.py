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

    def get_case_iter(self, dtype):
        ordinal = 0
        for shape in self.shapes:
            dims = [None]
            if len(shape) >= 2:
                dims.extend([-1, 0])
            if len(shape) >= 3:
                dims.append(1)
            for dim in dims:
                yield self._case_from_plan(
                    dtype,
                    ordinal,
                    base.BenchmarkCasePlan(
                        shape={"input": shape},
                        params={"dim": dim},
                        builder_args=(shape, dim),
                    ),
                )
                ordinal += 1

    def build_inputs(self, case):
        shape, dim = case.builder_args[0].builder_args
        x = torch.randn(shape, dtype=case.dtype, device=self.device) * 10
        mask = torch.rand(shape, device=self.device) > 0.7
        x[mask] = float("nan")
        return (x,) if dim is None else (x, dim)


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
