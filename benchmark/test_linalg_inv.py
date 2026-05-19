import pytest
import torch

from . import base


class LinalgInvBenchmark(base.Benchmark):
    def set_more_shapes(self):
        self.shapes = [(4, 2, 2), (16, 3, 3), (64, 2, 2), (256, 3, 3)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n = shape[-1]
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp = inp + torch.eye(n, dtype=cur_dtype, device=self.device) * 2.0
            yield {"A": inp}

    def get_golden_fn(self):
        return torch.linalg.inv

    def get_fn(self):
        return torch.linalg.inv


@pytest.mark.linalg_inv
def test_linalg_inv():
    bench = LinalgInvBenchmark(
        op_name="linalg_inv",
        torch_op=torch.linalg.inv,
        # linalg.inv does not support low precision dtypes (Half/BFloat16)
        dtypes=[torch.float32],
    )
    bench.run()
