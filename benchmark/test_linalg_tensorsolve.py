import pytest
import torch

from . import base


@pytest.mark.linalg_tensorsolve
def test_linalg_tensorsolve():
    # Simple benchmark for linalg.tensorsolve
    # A is (n, n), B is (n,), X is (n,)
    class LinalgTensorsolveBenchmark(base.Benchmark):
        def set_shapes(self, shape_file_path=None):
            # Use small shapes for this operation
            self.shapes = [(4,), (8,), (16,), (32,), (64,), (128,)]

        def set_more_shapes(self):
            return None

        def get_input_iter(self, cur_dtype):
            for n in self.shapes:
                A = torch.eye(n[0], dtype=cur_dtype, device=self.device)
                B = torch.randn(n[0], dtype=cur_dtype, device=self.device)
                yield A, B

    bench = LinalgTensorsolveBenchmark(
        op_name="linalg_tensorsolve",
        torch_op=torch.linalg.tensorsolve,
        # torch.linalg.tensorsolve does not support float16/bfloat16 on CUDA
        dtypes=[torch.float32],
    )
    bench.run()
