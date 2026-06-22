from typing import Generator

import pytest
import torch

from . import base, consts, utils


def adaptive_avg_pool3d_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # Common case: reduce to small output
    yield inp, {"output_size": (1, 1, 1)}
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # Half the spatial dimensions
        D, H, W = shape[-3], shape[-2], shape[-1]
        yield inp, {"output_size": (max(1, D // 2), max(1, H // 2), max(1, W // 2))}


class AdaptiveAvgPool3dBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        shapes_5d = [
            (4, 3, 16, 56, 56),
            (8, 64, 8, 28, 28),
            (16, 128, 4, 14, 14),
            (32, 256, 4, 7, 7),
        ]

        for shape in shapes_5d:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.adaptive_avg_pool3d
def test_perf_adaptive_avg_pool3d():
    bench = AdaptiveAvgPool3dBenchmark(
        input_fn=adaptive_avg_pool3d_input_fn,
        op_name="adaptive_avg_pool3d",
        torch_op=torch.ops.aten._adaptive_avg_pool3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def adaptive_avg_pool3d_out_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    out = torch.empty((shape[0], shape[1], 1, 1, 1), dtype=dtype, device=device)
    yield inp, (1, 1, 1), {"out": out}
    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        D, H, W = shape[-3], shape[-2], shape[-1]
        output_size = (max(1, D // 2), max(1, H // 2), max(1, W // 2))
        out = torch.empty(
            (shape[0], shape[1], *output_size), dtype=dtype, device=device
        )
        yield inp, output_size, {"out": out}


@pytest.mark.adaptive_avg_pool3d_out
def test_perf_adaptive_avg_pool3d_out():
    bench = AdaptiveAvgPool3dBenchmark(
        input_fn=adaptive_avg_pool3d_out_input_fn,
        op_name="adaptive_avg_pool3d_out",
        torch_op=torch.ops.aten._adaptive_avg_pool3d.out,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
