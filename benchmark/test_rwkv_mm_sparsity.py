import pytest
import torch

import flag_gems

from . import base, consts


# TODO(Qiming): Remove this class
class RWKVSparsityBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return []


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
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
