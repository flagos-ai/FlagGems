import pytest
import torch

import flag_gems

from . import attri_util as attr_utils
from . import performance_utils as utils


class RWKVSparsityBenchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        return None


def _input_fn(shape, dtype, device):
    n = 16384
    embedding_dim = 4096

    V_ = torch.randn(n, embedding_dim, dtype=dtype, device=device)
    sparsity_levels = [0.9]
    for target_sparsity in sparsity_levels:
        k_sparse = torch.randn(n, dtype=dtype, device=device)
        threshold = torch.quantile(
            k_sparse.abs().to(torch.float32), target_sparsity
        ).to(dtype)
        k_sparse = torch.relu(k_sparse - threshold)

        yield k_sparse, V_


def torch_rwkv_mm_sparsity(k, v):
    return torch.mv(v.T, k)


@pytest.mark.rwkv_mm_sparsity
def test_rwkv_mm_sparsity():
    bench = RWKVSparsityBenchmark(
        input_fn=_input_fn,
        op_name="rwkv_mm_sparsity",
        torch_op=torch_rwkv_mm_sparsity,
        gems_op=flag_gems.rwkv_mm_sparsity,
        dtypes=attr_utils.FLOAT_DTYPES,
    )

    bench.run()
