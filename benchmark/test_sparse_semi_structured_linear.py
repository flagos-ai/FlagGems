import pytest
import torch

from . import base, consts


class SparseSemiStructuredLinearBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def __init__(self, op_name, torch_op, use_bias=False, **kwargs):
        super().__init__(op_name, torch_op, dtypes=consts.FLOAT_DTYPES, **kwargs)
        self.use_bias = use_bias

    def set_shapes(self, shape_file_path=None):
        # Representative (M, K) shapes from small to medium matrix sizes
        self.shapes = [
            (16, 32),
            (64, 128),
            (256, 512),
        ]
        self.shape_desc = "M, K"

    def set_more_shapes(self):
        return [
            (128, 1024),
            (512, 2048),
        ]

    def get_input_iter(self, dtype):
        for M, K in self.shapes:
            N = K  # output features equal to input features
            input = torch.randn(M, K, dtype=dtype, device=self.device)
            weight = torch.randn(N, K, dtype=dtype, device=self.device)
            meta = torch.ones(N // 4, K, dtype=torch.int8, device=self.device)
            if self.use_bias:
                bias = torch.randn(N, dtype=dtype, device=self.device)
                yield input, weight, meta, bias
            else:
                yield input, weight, meta

    def get_tflops(self, op, *args, **kwargs):
        M, K = args[0].shape
        N = K
        return 2 * M * N * K


def _torch_ref(input, weight, meta):
    """Reference: dense matmul with meta=1 (all weights valid)."""
    return torch.matmul(input, weight.t())


def _torch_ref_with_bias(input, weight, meta, bias):
    return torch.matmul(input, weight.t()) + bias


@pytest.mark.sparse_semi_structured_linear
def test_sparse_semi_structured_linear():
    bench = SparseSemiStructuredLinearBenchmark(
        op_name="sparse_semi_structured_linear",
        torch_op=_torch_ref,
        use_bias=False,
    )
    bench.run()


@pytest.mark.sparse_semi_structured_linear
def test_sparse_semi_structured_linear_with_bias():
    bench = SparseSemiStructuredLinearBenchmark(
        op_name="sparse_semi_structured_linear",
        torch_op=_torch_ref_with_bias,
        use_bias=True,
    )
    bench.run()
