import pytest
import torch

import flag_gems

from . import base, consts

# Benchmark shapes covering a range of typical 3D grid sampling workloads
GRID_SAMPLER_3D_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 8, 16, 16, 16),
    (4, 16, 32, 32, 32),
    (8, 32, 64, 64, 64),
]


class GridSampler3DBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = GRID_SAMPLER_3D_SHAPES

    def build_inputs(self, case):
        if (
            flag_gems.vendor_name == "kunlunxin"
            and case.dtype in (torch.float16, torch.bfloat16)
            and "latency_base" in self.to_bench_metrics
            and not base.Config.skip_native
        ):
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlunxin native grid_sampler_3d supports float32 inputs only")
        return super().build_inputs(case)


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    grid = (dims[0], dims[2] // 2, dims[3] // 2, dims[4] // 2, 3)
    yield base.BenchmarkCasePlan(
        shape={"input": list(dims), "grid": list(grid)},
        params={"interpolation_mode": 0, "padding_mode": 0, "align_corners": False},
        builder_args=(dims, grid),
    )


def _build_inputs_fn(plan, dtype, device):
    shape, grid_shape = plan.builder_args
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    grid = torch.randn(grid_shape, dtype=dtype, device=device) * 1.5
    return inp, grid, 0, 0, False


@pytest.mark.grid_sampler_3d
def test_grid_sampler_3d():
    bench = GridSampler3DBenchmark(
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        op_name="grid_sampler_3d",
        torch_op=torch.ops.aten.grid_sampler_3d,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
