import pytest
import torch

from . import base, consts


def _make_sparse_coo(nnz, size, dtype, device):
    """Build a sparse COO tensor with unique, sorted indices so marking it
    ``coalesced`` is a valid assertion."""
    numel = 1
    for s in size:
        numel *= s
    nnz = min(nnz, numel)
    flat = torch.randperm(numel, device="cpu")[:nnz]
    nnd = len(size)
    indices = torch.empty((nnd, nnz), dtype=torch.int64)
    for d in reversed(range(nnd)):
        indices[d] = flat % size[d]
        flat = flat // size[d]
    perm = torch.arange(nnz)
    for d in reversed(range(nnd)):
        perm = perm[torch.argsort(indices[d][perm], stable=True)]
    indices = indices[:, perm].contiguous().to(device)
    values = torch.randn(nnz, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(indices, values, size)


# Sparse COO tensor shapes for the _coalesced_ benchmark: (nnz, size).
# The op only touches the coalesced metadata flag (plus a Triton identity
# pass over the values), so the relevant workload axis is the number of
# stored values (nnz).
SPARSE_COALESCED_SHAPES = [
    (8, (4, 4)),
    (1024, (32, 32)),
    (4096, (64, 64)),
    (16384, (128, 128)),
]


class CoalescedBenchmark(base.Benchmark):
    """Benchmark for ``aten::_coalesced_`` over sparse COO tensors.

    The operator is a metadata mutation backed by a Triton identity kernel
    over the values tensor, so the workload is dominated by the values size.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = SPARSE_COALESCED_SHAPES

    def get_input_iter(self, cur_dtype):
        for nnz, size in self.shapes:
            inp = _make_sparse_coo(nnz, size, cur_dtype, self.device)
            # ``aten::_coalesced_.default(self, coalesced)`` is a functional op.
            yield inp, True


@pytest.mark.coalesced_
def test_coalesced_():
    bench = CoalescedBenchmark(
        op_name="coalesced_",
        torch_op=torch.ops.aten._coalesced_.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
