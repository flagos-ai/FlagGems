import pytest
import torch

import flag_gems

from . import base, consts


class GatherSparseBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Keep the benchmark footprint bounded: the sparse output has one
        # non-zero per element of `index`, so the nnz (and thus the size of the
        # int64 indices and the values tensors) grows with the gather dim.
        self.shapes = [
            (1024, 1024),
            (4096, 4096),
            (8192, 8192),
            (256, 256, 256),
        ]

    def set_more_shapes(self):
        return []


def _gather_sparse_backward_input_fn(shape, dtype, device):
    """Yield ``(self, dim, index, grad)`` tuples for the gather sparse backward.

    `self` keeps the full dim while `index`/`grad` sample half of it along the
    gather dimension (a common real-world gradient shape).
    """
    inp = torch.randn(shape, dtype=dtype, device=device)
    for dim in range(len(shape)):
        index_shape = list(shape)
        index_shape[dim] = max(1, shape[dim] // 2)
        index = torch.randint(
            0, shape[dim], tuple(index_shape), dtype=torch.long, device=device
        )
        grad = torch.randn(tuple(index_shape), dtype=dtype, device=device)
        yield inp, dim, index, grad


@pytest.mark.gather_sparse_backward
@pytest.mark.skipif(
    (not torch.cuda.is_available()) or (flag_gems.device != "cuda"),
    reason="CUDA backend is not available for this benchmark.",
)
def test_gather_sparse_backward():
    bench = GatherSparseBackwardBenchmark(
        input_fn=_gather_sparse_backward_input_fn,
        op_name="gather_sparse_backward",
        torch_op=torch.ops.aten._gather_sparse_backward.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
