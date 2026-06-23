import pytest
import torch

from . import base


def kthvalue_input_fn(shape, dtype, device):
    x = torch.randn(shape, device=device, dtype=dtype)
    k = 2 if shape[-1] > 2 else shape[-1]
    yield {"input": x, "k": k, "dim": -1},


class KthvalueBenchmark(base.GenericBenchmarkExcluse1D):
    def set_shapes(self, shape_file_path=None):
        # 2D shapes for kthvalue along last dimension, exercising different dim sizes and batch sizes
        self.shapes = [
            (1024, 256),
            (4096, 64),
            (16384, 128),
            (512, 512),
            (2048, 512),
        ]


@pytest.mark.kthvalue
def test_kthvalue():
    bench = KthvalueBenchmark(
        op_name="kthvalue",
        torch_op=torch.kthvalue,
        # Benchmark uses float32 only because topk gemm kernel operates in float32;
        # the kthvalue op auto-converts non-fp32 inputs internally.
        dtypes=[torch.float32],
        input_fn=kthvalue_input_fn,
    )
    bench.run()
