import pytest
import torch

from . import base

# Typical shapes for loss backward benchmarks
BINARY_CROSS_ENTROPY_BACKWARD_SHAPES = [
    (1024,),
    (1024, 1024),
    (16, 128, 64),
    (16, 128, 64, 64),
]


class BinaryCrossEntropyBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = BINARY_CROSS_ENTROPY_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # self is probability in (0, 1) - required for BCE backward
            self_t = torch.rand(shape, dtype=cur_dtype, device=self.device)
            self_t = torch.clamp(self_t, 1e-4, 1 - 1e-4)
            target = torch.randint(0, 2, shape, dtype=cur_dtype, device=self.device).to(
                dtype=cur_dtype
            )
            grad_output = torch.ones_like(self_t)
            weight = torch.rand(shape, dtype=cur_dtype, device=self.device)
            # reduction=0 (none) for benchmarking
            yield grad_output, self_t, target, weight, 0


@pytest.mark.binary_cross_entropy_backward
def test_binary_cross_entropy_backward():
    bench = BinaryCrossEntropyBackwardBenchmark(
        op_name="binary_cross_entropy_backward",
        torch_op=torch.ops.aten.binary_cross_entropy_backward,
        # bfloat16 excluded: BCE backward requires precise (0,1) probability values
        dtypes=[torch.float16, torch.float32],
    )
    bench.run()
