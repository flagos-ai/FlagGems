import pytest
import torch

from . import base, consts

# Shapes for adaptive_avg_pool3d_backward (grad_input out-variant) benchmark
ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES = [
    (1, 3, 8, 8, 8),
    (2, 3, 16, 16, 16),
    (1, 1, 32, 32, 32),
    (4, 8, 64, 64, 64),
]


class AdaptiveAvgPool3DBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ADAPTIVE_AVG_POOL3D_BACKWARD_SHAPES
        self.output_sizes = [(4, 4, 4), (8, 8, 8), (16, 16, 16), (32, 32, 32)]

    def get_case_iter(self, cur_dtype):
        for ordinal, (shape, output_size) in enumerate(
            zip(self.shapes, self.output_sizes)
        ):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": list(shape)},
                    params={"output_size": list(output_size)},
                    builder_args=(shape, output_size),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        shape, output_size = plan.builder_args
        cur_dtype = case.dtype
        x = torch.randn(shape, dtype=cur_dtype, device=self.device)
        # The gradient shape is known; building it does not require a forward kernel.
        grad = torch.ones(
            (*shape[:-3], *output_size), dtype=cur_dtype, device=self.device
        )
        # The out-variant writes into a caller-provided buffer.
        grad_input = torch.empty_like(x)
        return grad, x, {"grad_input": grad_input}


@pytest.mark.adaptive_avg_pool3d_backward
@pytest.mark.adaptive_avg_pool3d_backward_grad_input
def test_adaptive_avg_pool3d_backward():
    bench = AdaptiveAvgPool3DBackwardBenchmark(
        op_name="adaptive_avg_pool3d_backward_grad_input",
        torch_op=torch.ops.aten.adaptive_avg_pool3d_backward.grad_input,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
