import pytest
import torch

from . import base, consts

# Benchmark shapes covering 2D/3D inputs with various padding combinations
REPLICATION_PAD1D_BACKWARD_SHAPES = [
    ((2, 3, 7), (1, 2)),
    ((4, 16, 64), (3, 1)),
    ((8, 32, 256), (1, 2)),
    ((32, 256), (3, 1)),
]


class ReplicationPad1dBackwardBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = REPLICATION_PAD1D_BACKWARD_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape, padding in self.shapes:
            W_in = shape[-1]
            W_out = W_in + padding[0] + padding[1]
            grad_output = torch.randn(1, 1, W_out, dtype=cur_dtype, device=self.device)
            self_tensor = torch.randn(1, 1, W_in, dtype=cur_dtype, device=self.device)
            yield grad_output, self_tensor, list(padding)


@pytest.mark.replication_pad1d_backward
def test_replication_pad1d_backward():
    bench = ReplicationPad1dBackwardBenchmark(
        op_name="replication_pad1d_backward",
        torch_op=torch.ops.aten.replication_pad1d_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
