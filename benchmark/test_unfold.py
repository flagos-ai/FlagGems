import pytest
import torch

import flag_gems

from . import performance_utils as utils


def _input_fn(config, dtype, device):
    input_sizes, dim, size, step = config
    d = dim % len(input_sizes)
    num_windows = (input_sizes[d] - size) // step + 1
    grad_shape = (
        list(input_sizes[:d]) + [num_windows] + list(input_sizes[d + 1 :]) + [size]
    )
    grad_in = torch.randn(grad_shape, dtype=dtype, device=device)
    yield grad_in, list(input_sizes), dim, size, step


class UnfoldBackwardBenchmark(utils.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            ((32, 64), 1, 16, 16),
            ((16, 33), 0, 5, 2),
            ((4, 8, 12), -1, 6, 4),
            ((7, 13), 1, 13, 3),
            ((6, 20), 1, 7, 4),
            ((2, 3, 17), -1, 9, 1),
            ((2, 17), 1, 4, 6),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for config in self.shapes:
            yield from _input_fn(config, cur_dtype, self.device)


@pytest.mark.unfold_backward
def test_unfold_backward():
    bench = UnfoldBackwardBenchmark(
        op_name="unfold_backward",
        torch_op=torch.ops.aten.unfold_backward,
        dtypes=[torch.float16, torch.float32, torch.bfloat16],
    )
    bench.set_gems(flag_gems.unfold_backward)
    bench.run()
