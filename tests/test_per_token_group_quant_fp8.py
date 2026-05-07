import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def native_per_token_group_quant_fp8(
    x, group_size, eps=1e-10, dtype=None, scale_ue8m0=False
):
    if dtype is None:
        dtype = flag_gems.SUPPORTED_FP8_DTYPE

    assert (
        x.shape[-1] % group_size == 0
    ), "the last dimension of `x` cannot be divisible by `group_size`"
    assert x.is_contiguous(), "`x` is not contiguous"

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max

    x_ = x.reshape(x.numel() // group_size, group_size)
    amax = x_.abs().max(dim=-1, keepdim=True)[0].clamp(min=eps).to(torch.float32)
    x_s = amax * torch.tensor(1.0 / fp8_max, dtype=torch.float32, device=x.device)
    if scale_ue8m0:
        min_val = torch.tensor(1e-10, dtype=x_s.dtype, device=x_s.device)
        x_s = torch.exp2(torch.ceil(torch.log2(torch.maximum(x_s.abs(), min_val))))
    x_q = (x_ / x_s).clamp(min=fp8_min, max=fp8_max).to(dtype)
    x_q = x_q.reshape(x.shape)
    x_s = x_s.reshape(x.shape[:-1] + (x.shape[-1] // group_size,))

    return x_q, x_s


@pytest.mark.per_token_group_quant_fp8
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("group_size", utils.FP8_QUANT_SHAPES["GROUP_SIZE"])
@pytest.mark.parametrize("dtype", utils.FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
@pytest.mark.parametrize("scale_ue8m0", [True, False])
def test_per_token_group_quant_fp8(num_tokens, d, dtype, group_size, seed, scale_ue8m0):
    torch.manual_seed(seed)

    x = torch.rand(num_tokens, d, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out, ref_scale = native_per_token_group_quant_fp8(
        ref_x, group_size, scale_ue8m0=scale_ue8m0
    )
    with flag_gems.use_gems():
        out, scale = flag_gems.per_token_group_quant_fp8(
            x, group_size, scale_ue8m0=scale_ue8m0
        )

    utils.gems_assert_close(scale, ref_scale, dtype=torch.float32)

    out_fp32 = utils.to_cpu(out, ref_out).to(torch.float32)
    ref_out_fp32 = ref_out.to(torch.float32)

    assert torch.allclose(out_fp32, ref_out_fp32, rtol=0.15)
