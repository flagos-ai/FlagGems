import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def native_per_token_group_quant_int8(x, group_size, eps=1e-10):
    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    int8_min = torch.iinfo(torch.int8).min
    int8_max = torch.iinfo(torch.int8).max
    x_ = x.float().reshape(x.numel() // group_size, group_size)
    scale = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps)
    scale = scale / int8_max
    x_q = (x_ / scale).clamp(min=int8_min, max=int8_max).to(torch.int8)
    x_q = x_q.reshape(x.shape)
    scale = scale.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))
    return x_q, scale


@pytest.mark.per_token_group_quant_int8
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("group_size", utils.FP8_QUANT_SHAPES["GROUP_SIZE"])
@pytest.mark.parametrize("dtype", utils.FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
def test_per_token_group_quant_int8(num_tokens, d, dtype, group_size, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype, device=flag_gems.device).contiguous()
    ref_x = utils.to_reference(x)

    ref_out, ref_scale = native_per_token_group_quant_int8(ref_x, group_size)
    with flag_gems.use_gems():
        out, scale = flag_gems.per_token_group_quant_int8(x, group_size)

    utils.gems_assert_close(scale, ref_scale, dtype=torch.float32)
    utils.gems_assert_close(out, ref_out, dtype=torch.int8, atol=1)
