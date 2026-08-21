import pytest
import torch

import flag_gems

from . import base

# (M, K, N) shapes for matmul_backward benchmark: out = (M, K) @ (K, N)
MATMUL_BACKWARD_SHAPES = [
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
]


class MatmulBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = MATMUL_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for m, k, n in self.shapes:
            self_t = torch.randn(m, k, dtype=cur_dtype, device=self.device)
            other_t = torch.randn(k, n, dtype=cur_dtype, device=self.device)
            grad = torch.randn(m, n, dtype=cur_dtype, device=self.device)
            yield grad, self_t, other_t, [True, True]


@pytest.mark.matmul_backward
def test_matmul_backward():
    bench = MatmulBackwardBenchmark(
        op_name="matmul_backward",
        # No native PyTorch CUDA impl for aten::matmul_backward, so use the
        # flag_gems implementation as both baseline and gems op.
        torch_op=flag_gems.matmul_backward,
        # Restricted dtype list per the worktree generator (numerical-stability / precision scope).
        dtypes=[torch.float32, torch.float16],
    )
    bench.set_gems(flag_gems.matmul_backward)
    bench.run()
