import random
import time

import numpy as np
import pytest
import torch

import flag_gems
from flag_gems.ops.topk import topk_fp8_w8a16

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@pytest.mark.topk
@pytest.mark.parametrize("batch_size", [4, 8])
@pytest.mark.parametrize("hiddensize", [128, 256])
@pytest.mark.parametrize("topk", [0, 5])
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk(batch_size, hiddensize, topk, largest, dtype):
    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    # Each row use different shuffled index.
    for bsz in range(batch_size):
        col_indices = torch.randperm(x.size(1))
        x[bsz, :] = x[bsz, col_indices]
    ref_x = utils.to_reference(x)

    # Bug #2856
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        ref_x = ref_x.cuda()

    ref_value, ref_index = torch.topk(ref_x, topk, largest=largest)

    # Bug #2856
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float16:
        if cfg.TO_CPU:
            ref_value = ref_value.cpu()
            ref_index = ref_index.cpu()

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, largest=largest)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.topk
@pytest.mark.parametrize(
    "shape, topk",
    [
        ((16, 1024, 256), 256),
        ((8, 512, 32), 32),
        ((4, 128, 64), 64),
        ((2, 33, 128), 128),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk_3d_lastdim(shape, topk, dtype):
    batch_size = int(np.prod(shape[:-1]))
    hiddensize = shape[-1]

    x = torch.arange(hiddensize, dtype=dtype, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(shape)
    x_2d = x.reshape(batch_size, hiddensize)

    for bsz in range(batch_size):
        col_indices = torch.randperm(hiddensize)
        x_2d[bsz, :] = x_2d[bsz, col_indices]

    ref_x = utils.to_reference(x)
    ref_value, ref_index = torch.topk(ref_x, topk, dim=-1, largest=True, sorted=True)

    with flag_gems.use_gems():
        res_value, res_index = torch.topk(x, topk, dim=-1, largest=True, sorted=True)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


GROUP_SIZE = 128
FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)


def _quantize_fp8_grouped_lastdim(x, group_size=GROUP_SIZE):
    fp8_info = torch.finfo(FP8_DTYPE)
    n = x.shape[-1]
    num_groups = (n + group_size - 1) // group_size
    padded_n = num_groups * group_size
    if padded_n != n:
        pad_shape = x.shape[:-1] + (padded_n - n,)
        x_for_scale = torch.cat(
            [x, torch.zeros(pad_shape, device=x.device, dtype=x.dtype)], dim=-1
        )
    else:
        x_for_scale = x
    grouped = x_for_scale.reshape(*x.shape[:-1], num_groups, group_size).float()
    scale = (grouped.abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
        min=1e-8
    )
    q = (grouped / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    q = q.reshape(*x.shape[:-1], padded_n)[..., :n].contiguous()
    return q, scale.squeeze(-1).to(x.dtype).contiguous()


def _dequant_fp8_grouped_lastdim(x_fp8, x_scale, group_size=GROUP_SIZE):
    n = x_fp8.shape[-1]
    group_ids = torch.arange(n, device=x_fp8.device) // group_size
    return x_fp8.float() * x_scale.index_select(-1, group_ids).float()


def _quantize_fp8_row(x):
    fp8_info = torch.finfo(FP8_DTYPE)
    scale = (x.float().abs().amax(dim=-1, keepdim=True) / fp8_info.max).clamp(
        min=1e-8
    )
    q = (x.float() / scale).clamp(fp8_info.min, fp8_info.max).to(FP8_DTYPE)
    return q.contiguous(), scale.to(x.dtype).contiguous()


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 TopK requires CUDA FP8 support",
)
@pytest.mark.parametrize("shape, topk", [((4, 128), 5), ((8, 256), 16), ((2, 1024), 8)])
@pytest.mark.parametrize("largest", [True, False])
def test_topk_fp8_w8a16(shape, topk, largest):
    x = torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_grouped_lastdim(x)
    x_dequant = _dequant_fp8_grouped_lastdim(x_fp8, x_scale)

    ref_value, ref_index = torch.topk(
        x_dequant, topk, dim=-1, largest=largest, sorted=True
    )
    res_value, res_index = topk_fp8_w8a16(
        x_fp8, x_scale, topk, dim=-1, largest=largest, sorted=True
    )

    gathered = torch.gather(x_dequant, dim=-1, index=res_index)
    torch.testing.assert_close(res_value.float(), ref_value, rtol=0, atol=2e-2)
    torch.testing.assert_close(gathered, res_value.float(), rtol=0, atol=2e-2)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.device != "cuda" or not hasattr(torch, "float8_e4m3fn"),
    reason="FP8 TopK requires CUDA FP8 support",
)
@pytest.mark.parametrize("shape, topk", [((4, 128), 5), ((8, 256), 16), ((2, 4096), 8)])
def test_topk_fp8_w8a16_row_scale(shape, topk):
    x = torch.randn(shape, dtype=torch.bfloat16, device=flag_gems.device)
    x_fp8, x_scale = _quantize_fp8_row(x)
    x_dequant = x_fp8.float() * x_scale.float()

    ref_value, ref_index = torch.topk(x_dequant, topk, dim=-1, largest=True, sorted=True)
    res_value, res_index = topk_fp8_w8a16(
        x_fp8,
        x_scale,
        topk,
        dim=-1,
        largest=True,
        sorted=True,
        group_size=shape[-1],
    )

    gathered = torch.gather(x_dequant, dim=-1, index=res_index)
    torch.testing.assert_close(res_value.float(), ref_value, rtol=0, atol=2e-2)
    torch.testing.assert_close(gathered, res_value.float(), rtol=0, atol=2e-2)
