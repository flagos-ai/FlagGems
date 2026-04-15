import pytest
import torch

import flag_gems

from . import accuracy_utils as utils


def native_dynamic_scaled_fp8_quant(x, eps=1e-10):
    assert x.ndim == 2 and x.stride(-1) == 1
    fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
    fp8_min = float(torch.finfo(torch.float8_e4m3fn).min)
    min_scale = 1.0 / (fp8_max * 512.0)

    scale = (x.abs().max().clamp(min=eps).to(torch.float32) / fp8_max).clamp(
        min=min_scale
    )
    x_q = (x.float() / scale).clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)
    return x_q, scale.reshape(1)


@pytest.mark.dynamic_scaled_fp8_quant
@pytest.mark.parametrize("seed", utils.FP8_QUANT_SHAPES["SEEDS"])
@pytest.mark.parametrize("dtype", utils.FP8_QUANT_SHAPES["DTYPES"])
@pytest.mark.parametrize("d", utils.FP8_QUANT_SHAPES["D"])
@pytest.mark.parametrize("num_tokens", utils.FP8_QUANT_SHAPES["NUM_TOKENS"])
def test_dynamic_scaled_fp8_quant(num_tokens, d, dtype, seed):
    torch.manual_seed(seed)
    x = torch.rand(num_tokens, d, dtype=dtype, device=flag_gems.device)
    ref_x = utils.to_reference(x)

    ref_out, ref_scale = native_dynamic_scaled_fp8_quant(ref_x)
    with flag_gems.use_gems():
        out, scale = flag_gems.dynamic_scaled_fp8_quant(x)

    utils.gems_assert_close(scale, ref_scale, dtype=torch.float32)

    out_fp32 = utils.to_cpu(out, ref_out).to(torch.float32)
    ref_out_fp32 = ref_out.to(torch.float32)
    assert torch.allclose(out_fp32, ref_out_fp32, rtol=0.15)
