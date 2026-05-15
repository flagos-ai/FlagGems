import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


@pytest.mark.rwkv_ka_fusion
@pytest.mark.parametrize("T", [2**d for d in range(4, 15, 2)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_accuracy_rwkv_kafusion(T, dtype):
    H = 8
    N = 64
    C = H * N
    k = torch.rand(T, C, dtype=dtype, device=flag_gems.device)
    kk = torch.rand(C, dtype=dtype, device=flag_gems.device)
    a = torch.rand(T, C, dtype=dtype, device=flag_gems.device)
    ka = torch.rand(C, dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        o_k, o_kk, o_kka = flag_gems.rwkv_ka_fusion(k, kk, a, ka, H, N)

    ref_k = utils.to_reference(k, True)
    ref_kk = utils.to_reference(kk, True)
    ref_a = utils.to_reference(a, True)
    ref_ka = utils.to_reference(ka, True)

    ref_o_kk = torch.nn.functional.normalize(
        (ref_k * ref_kk).view(T, H, N), dim=-1, p=2.0
    ).view(T, H * N)
    ref_o_k = ref_k * (1 + (ref_a - 1) * ref_ka)
    ref_o_kka = ref_o_kk * ref_a

    utils.gems_assert_close(o_k, ref_o_k, dtype, equal_nan=True)
    utils.gems_assert_close(o_kk, ref_o_kk, dtype, equal_nan=True)
    utils.gems_assert_close(o_kka, ref_o_kka, dtype, equal_nan=True)
