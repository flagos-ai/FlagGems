import pytest
import torch

import flag_gems

from . import base, consts


class GatherSparseBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [(1024, 1024), (4096, 4096), (8192, 8192), (256, 256, 256)]

    def set_more_shapes(self):
        return []


def _case_fn(shape, dtype):
    del dtype
    dims = tuple(shape)
    for dim in range(len(dims)):
        index_shape = list(dims)
        index_shape[dim] = max(1, dims[dim] // 2)
        yield base.BenchmarkCasePlan(
            shape={"self": list(dims), "index": index_shape, "grad": index_shape},
            params={"dim": dim},
            builder_args=(dims, tuple(index_shape)),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, index_shape = plan.builder_args
    dim = plan.params["dim"]
    inp = torch.randn(shape, dtype=dtype, device=device)
    index = torch.randint(0, shape[dim], index_shape, dtype=torch.long, device=device)
    grad = torch.randn(index_shape, dtype=dtype, device=device)
    return inp, dim, index, grad


@pytest.mark.gather_sparse_backward
@pytest.mark.underscore_gather_sparse_backward
@pytest.mark.skipif((not torch.cuda.is_available()) or (flag_gems.device != "cuda"), reason="CUDA backend is not available for this benchmark.")
def test_gather_sparse_backward():
    GatherSparseBackwardBenchmark(
        op_name="_gather_sparse_backward",
        torch_op=torch.ops.aten._gather_sparse_backward.default,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=consts.FLOAT_DTYPES,
    ).run()
