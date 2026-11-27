import pytest
import torch

import flag_gems

from . import base, consts


def _decomposed_nanmean(x, dim=None):
    """Baseline for enflame: nan-aware mean built from aten primitives.

    torch.nanmean and torch.nansum both segfault on GCU as soon as the input
    carries NaN (measured on a 1M FP32 tensor), so the baseline sums the non-NaN
    values and divides by their count instead. That is the same arithmetic
    without any fused nan-aware kernel, which is exactly what a baseline leg
    should measure.
    """
    reduced_count = x.numel() if dim is None else x.shape[dim % x.ndim]
    nan_mask = torch.isnan(x)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    total = torch.where(nan_mask, zero, x).sum(dim=dim)
    valid = float(reduced_count) - nan_mask.sum(dim=dim).to(x.dtype)
    return total / valid


def _decomposed_nanmean_out(x, dim=None, *, out=None):
    value = _decomposed_nanmean(x, dim)
    out.resize_(value.shape)
    out.copy_(value)
    return out


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
                    # The reduced extent is the output shape for keepdim=False; ask
                    # the shape directly so the setup never runs aten nanmean, which
                    # segfaults on GCU for NaN-bearing input.
                    out_shape = list(x.shape)
                    del out_shape[dim % x.ndim]
                    out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                    yield x, dim, {"out": out}

            if x.ndim >= 3:
                out_shape = list(x.shape)
                del out_shape[1]
                out = torch.empty(out_shape, dtype=cur_dtype, device=self.device)
                yield x, 1, {"out": out}


@pytest.mark.nanmean
def test_benchmark_nanmean():
    torch_op = (
        _decomposed_nanmean if flag_gems.vendor_name == "enflame" else torch.nanmean
    )
    bench = _NanmeanBenchmark(
        op_name="nanmean",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.nanmean_out
def test_benchmark_nanmean_out():
    torch_op = (
        _decomposed_nanmean_out if flag_gems.vendor_name == "enflame" else torch.nanmean
    )
    bench = _NanmeanOutBenchmark(
        op_name="nanmean_out",
        torch_op=torch_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.nanmean_out)
    bench.run()
