import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rwkv_mm_sparsity
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_rwkv_mmsparsity(dtype):
    n = 16384
    embedding_dim = 4096

    k = torch.randn(n, dtype=dtype, device=flag_gems.device)
    k = torch.relu(k)
    if flag_gems.vendor_name == "kunlunxin":
        torch.manual_seed(42)
        # kunlunxin sparsity test require 90% sparsity
        sparsity_levels = [0.9]
        for target_sparsity in sparsity_levels:
            threshold = torch.quantile(k.abs().to(torch.float32), target_sparsity).to(
                dtype
            )
            k = torch.relu(k - threshold)

    V_ = torch.randn(n, embedding_dim, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        res = flag_gems.rwkv_mm_sparsity(k, V_)

    ref_k = utils.to_reference(k, True)
    ref_V_ = utils.to_reference(V_, True)
    ref_res = ref_k @ ref_V_

    utils.gems_assert_close(res, ref_res, dtype, equal_nan=True)
