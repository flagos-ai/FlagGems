import pytest
import torch

import flag_gems

from . import base

INVERSE_DTYPES = [torch.float32] + (
    [torch.float64] if flag_gems.runtime.device.support_fp64 else []
)
INVERSE_SHAPES = [
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 128),
    (256, 256),
    (1024, 8, 8),
    (512, 16, 16),
    (128, 32, 32),
    (32, 64, 64),
    (8, 128, 128),
]


class InverseBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = INVERSE_SHAPES
        self.shape_desc = "batch..., N, N"

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={},
                    builder_args=(tuple(shape),),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        needs_native = getattr(base.Config, "reference_only", False) or (
            "latency_base" in self.to_bench_metrics and not base.Config.skip_native
        )
        if flag_gems.vendor_name == "kunlunxin" and shape[-1] >= 64 and needs_native:
            base.Config.skipped_case_ids.add(case.case_id)
            pytest.skip("Kunlun native inverse reference rejects matrix sides >=64")
        torch.manual_seed(0)
        n = shape[-1]
        identity = torch.eye(n, dtype=case.dtype, device=self.device)
        if len(shape) > 2:
            identity = identity.expand(shape).contiguous()
        matrix = identity + 0.05 * torch.randn(
            shape, dtype=case.dtype, device=self.device
        )
        return (matrix,)


@pytest.mark.inverse
def test_inverse():
    bench = InverseBenchmark(
        op_name="inverse",
        torch_op=torch.inverse,
        dtypes=INVERSE_DTYPES,
    )
    bench.set_gems(flag_gems.inverse)
    bench.run()
