# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import time
from importlib import import_module

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

random.seed(time.time() // 100)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend overrides flag_gems.topk with the DSA-only op (no largest/dim/sorted); see test_topk_ascend_dsa",
)
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

    res_value, res_index = flag_gems.topk(x, topk, largest=largest)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend overrides flag_gems.topk with the DSA-only op (no largest/dim/sorted); see test_topk_ascend_dsa",
)
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

    res_value, res_index = flag_gems.topk(x, topk, dim=-1, largest=True, sorted=True)

    utils.gems_assert_close(res_value, ref_value, dtype)
    utils.gems_assert_equal(res_index, ref_index)


@pytest.mark.topk
def test_topk_radix_tle_config_uses_large_fp32_shape_heuristic():
    topk_op = import_module(flag_gems.topk.__module__)

    assert topk_op._get_topk_radix_tle_config(torch.float32, 32768, 256, 256) == (
        1024,
        8,
        8,
    )
    assert topk_op._get_topk_radix_tle_config(torch.float16, 32768, 256, 256) == (
        512,
        4,
        4,
    )
    assert topk_op._get_topk_radix_tle_config(torch.float32, 8192, 128, 128) == (
        512,
        4,
        4,
    )
    assert topk_op._get_topk_radix_tle_config(torch.float32, 32768, 512, 512) == (
        512,
        4,
        4,
    )
    assert topk_op._get_topk_radix_tle_config(torch.float32, 32768, 129, 256) == (
        512,
        4,
        4,
    )


@pytest.mark.topk
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_topk_radix_tle_large_fp32_k256_correctness():
    topk_op = import_module(flag_gems.topk.__module__)
    if not topk_op.HAS_TLE:
        pytest.skip("TLE topk path is unavailable")

    batch_size = 2
    hiddensize = 32768
    topk = 256
    x = torch.arange(hiddensize, dtype=torch.float32, device=flag_gems.device)
    x = x.repeat(batch_size).reshape(batch_size, hiddensize)

    ref_value, ref_index = torch.topk(
        utils.to_reference(x), topk, dim=-1, largest=True, sorted=True
    )

    res_value, res_index = flag_gems.topk(x, topk, dim=-1, largest=True, sorted=True)

    utils.gems_assert_close(res_value, ref_value, torch.float32)
    utils.gems_assert_equal(res_index, ref_index)


# Ascend DSA topk (ported from FlagTree): 2D fp32, largest=True only, direct
# call without use_gems (torch.topk stays dispatched to torch_npu).
@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend", reason="Ascend DSA topk required"
)
@pytest.mark.parametrize(
    "m, n, k",
    [
        (4, 128, 5),
        (64, 64, 5),
        (10000, 256, 5),
        (8, 4096, 64),
        (2, 4096, 128),
        (4, 8192, 512),
        (64, 16384, 256),
        (8, 32768, 4096),
        (128, 131072, 4096),
    ],
)
def test_topk_ascend_dsa(m, n, k):
    from packaging.version import Version
    from torch_npu.npu.utils import get_cann_version

    if Version(get_cann_version()) < Version("9.1.0"):
        pytest.skip("Ascend DSA topk requires CANN >= 9.1.0")

    # Random permutation of 0..n-1 per row: values are unique, so topk has no
    # ties and the index order is deterministic across implementations.
    x = torch.argsort(torch.rand((m, n), device=flag_gems.device), dim=1).to(
        torch.float32
    )
    ref_value, ref_index = torch.topk(utils.to_reference(x), k, dim=-1)

    res_value, res_index = flag_gems.topk(x, k)

    utils.gems_assert_close(res_value, ref_value, torch.float32)
    utils.gems_assert_equal(res_index.to(torch.int64), ref_index)
    # Consistency: values must match the input at the returned indices.
    utils.gems_assert_equal(
        x.gather(1, res_index.to(torch.int64)),
        utils.to_reference(res_value.to(torch.float32)),
    )


def _assert_topk_matches_reference(x_cpu, res_val, res_idx, k, dim, largest):
    ref_val, ref_idx = torch.topk(x_cpu, k, dim=dim, largest=largest, sorted=True)
    res_val_cpu = res_val.cpu()
    res_idx_cpu = res_idx.cpu()
    assert res_val_cpu.shape == ref_val.shape
    assert res_idx_cpu.shape == ref_idx.shape
    assert res_idx_cpu.dtype == torch.int64
    assert res_val.is_contiguous()
    assert res_idx.is_contiguous()
    torch.testing.assert_close(res_val_cpu, ref_val, rtol=0, atol=0)
    if k > 0 and x_cpu.numel() > 0:
        dim_size = x_cpu.shape[dim]
        assert bool(((res_idx_cpu >= 0) & (res_idx_cpu < dim_size)).all())
        gathered = torch.gather(x_cpu, dim, res_idx_cpu)
        torch.testing.assert_close(gathered, res_val_cpu, rtol=0, atol=0)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend overrides flag_gems.topk with the DSA-only op",
)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk_transposed_2d_repro(largest, dtype):
    torch.manual_seed(0)
    x_cpu = torch.randn(128, 128, dtype=torch.float32).to(dtype)
    x_dev = x_cpu.to(flag_gems.device)
    x_cpu_t = x_cpu.transpose(0, 1)
    x_dev_t = x_dev.transpose(0, 1)
    assert not x_dev_t.is_contiguous()
    assert x_dev_t.stride() == (1, 128)

    res_val, res_idx = flag_gems.topk(x_dev_t, 8, dim=-1, largest=largest, sorted=True)
    _assert_topk_matches_reference(x_cpu_t, res_val, res_idx, 8, -1, largest)

    # Issue #7829: distinct linspace values with stride (1, 2) verify exact index parity too.
    lin_cpu = (
        torch.linspace(-12, 13, 128, dtype=dtype, device="cpu")
        .reshape(2, 64)
        .t()
        .contiguous()
        .t()
    )
    lin_dev = lin_cpu.t().contiguous().to(flag_gems.device).t()
    assert lin_dev.stride() == (1, 2)
    for k in (1, 7, 64):
        v_res, i_res = flag_gems.topk(lin_dev, k, dim=-1, largest=largest, sorted=True)
        _assert_topk_matches_reference(lin_cpu, v_res, i_res, k, -1, largest)
        if dtype == torch.float32:
            _, i_ref = torch.topk(lin_cpu, k, dim=-1, largest=largest, sorted=True)
            torch.testing.assert_close(i_res.cpu(), i_ref, rtol=0, atol=0)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend overrides flag_gems.topk with the DSA-only op",
)
@pytest.mark.parametrize("largest", [True, False])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_topk_noncontiguous_dims_and_offsets(largest, dtype):
    # 1D sliced view with non-zero storage_offset and step > 1
    base_1d = torch.arange(220, dtype=torch.float32).sin().to(dtype)
    view_1d_cpu = base_1d[7:199:3]
    view_1d_dev = base_1d.to(flag_gems.device)[7:199:3]
    assert view_1d_dev.storage_offset() == 7
    assert view_1d_dev.stride() == (3,)
    for k in (0, 1, 9, view_1d_cpu.numel()):
        res_v, res_i = flag_gems.topk(
            view_1d_dev, k, dim=-1, largest=largest, sorted=True
        )
        _assert_topk_matches_reference(view_1d_cpu, res_v, res_i, k, -1, largest)

    # 2D sliced + transposed view across dim=0, dim=1, dim=-1
    torch.manual_seed(11)
    base_2d = torch.randn(40, 90, dtype=torch.float32).to(dtype)
    view_2d_cpu = base_2d[3:35:2, 5:77:3].transpose(0, 1)
    view_2d_dev = base_2d.to(flag_gems.device)[3:35:2, 5:77:3].transpose(0, 1)
    assert view_2d_dev.storage_offset() > 0
    assert not view_2d_dev.is_contiguous()
    for dim in (0, 1, -1):
        res_v, res_i = flag_gems.topk(
            view_2d_dev, 6, dim=dim, largest=largest, sorted=True
        )
        _assert_topk_matches_reference(view_2d_cpu, res_v, res_i, 6, dim, largest)

    # 3D permuted + sliced view across dim=0, dim=1, dim=-1
    torch.manual_seed(22)
    base_3d = torch.randn(7, 19, 13, dtype=torch.float32).to(dtype)
    view_3d_cpu = base_3d[1:6, 2:18:2, 1:12].permute(2, 0, 1)
    view_3d_dev = base_3d.to(flag_gems.device)[1:6, 2:18:2, 1:12].permute(2, 0, 1)
    assert view_3d_dev.storage_offset() > 0
    for dim in (0, 1, -1):
        res_v, res_i = flag_gems.topk(
            view_3d_dev, 4, dim=dim, largest=largest, sorted=True
        )
        _assert_topk_matches_reference(view_3d_cpu, res_v, res_i, 4, dim, largest)

    # 4D permuted + sliced view across dim=0, dim=1, dim=2, dim=-1
    torch.manual_seed(33)
    base_4d = torch.randn(4, 6, 9, 11, dtype=torch.float32).to(dtype)
    view_4d_cpu = base_4d[1:4, :, 1:8:2, 2:10].permute(2, 0, 3, 1)
    view_4d_dev = base_4d.to(flag_gems.device)[1:4, :, 1:8:2, 2:10].permute(2, 0, 3, 1)
    assert view_4d_dev.storage_offset() > 0
    for dim in (0, 1, 2, -1):
        k = min(3, view_4d_cpu.shape[dim])
        res_v, res_i = flag_gems.topk(
            view_4d_dev, k, dim=dim, largest=largest, sorted=True
        )
        _assert_topk_matches_reference(view_4d_cpu, res_v, res_i, k, dim, largest)


@pytest.mark.topk
@pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason="Ascend overrides flag_gems.topk with the DSA-only op",
)
@pytest.mark.parametrize("largest", [True, False])
def test_topk_strided_multistage_and_infinities(largest):
    # Multi-stage (>= 4096) transposed view with both small k and large k (chunk_size < k -> 2048)
    torch.manual_seed(44)
    base_large = torch.randn(4120, 4, dtype=torch.float32)
    x_cpu = base_large[8:4116, :].transpose(0, 1)  # shape (4, 4108), stride (1, 4)
    x_dev = base_large.to(flag_gems.device)[8:4116, :].transpose(0, 1)
    assert x_dev.shape == (4, 4108)
    assert x_dev.stride() == (1, 4)
    for k in (8, 256, 1200):
        res_v, res_i = flag_gems.topk(x_dev, k, dim=-1, largest=largest, sorted=True)
        _assert_topk_matches_reference(x_cpu, res_v, res_i, k, -1, largest)

    # Multi-stage along non-last dim (dim=1) on a 3D non-contiguous view
    base_3d_large = torch.randn(2, 4105, 3, dtype=torch.float32)
    res_v, res_i = flag_gems.topk(
        base_3d_large.to(flag_gems.device), 16, dim=1, largest=largest, sorted=True
    )
    _assert_topk_matches_reference(base_3d_large, res_v, res_i, 16, 1, largest)

    # Non-power-of-two strided view containing +/-inf values (padding-index safety)
    fill = float("-inf") if largest else float("inf")
    inf_base = torch.full((100, 3), fill, dtype=torch.float32)
    inf_base[0, :] = 0.3
    inf_base[1, :] = 2.0
    inf_base[2, :] = 1.0
    inf_base[3, :] = float("inf") if largest else float("-inf")
    inf_cpu = inf_base.transpose(0, 1)  # shape (3, 100), stride (1, 3)
    inf_dev = inf_base.to(flag_gems.device).transpose(0, 1)
    res_v, res_i = flag_gems.topk(inf_dev, 100, dim=-1, largest=largest, sorted=True)
    _assert_topk_matches_reference(inf_cpu, res_v, res_i, 100, -1, largest)
    for row in range(3):
        assert res_i[row].cpu().unique().numel() == 100
