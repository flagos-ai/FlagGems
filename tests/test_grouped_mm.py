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

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

GNK_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


def _make_grouped_inputs(groups, n_size, k_size, dtype):
    torch.manual_seed(0)
    group_sizes = [1 + (group_idx * 17) % 31 for group_idx in range(groups)]
    total_m = sum(group_sizes)
    mat_a = torch.rand((total_m, k_size), dtype=dtype, device=flag_gems.device)
    mat_b = torch.rand((groups, k_size, n_size), dtype=dtype, device=flag_gems.device)
    offs = torch.tensor(
        [sum(group_sizes[: group_idx + 1]) for group_idx in range(groups)],
        dtype=torch.int32,
        device=flag_gems.device,
    )
    return mat_a, mat_b, offs


def _grouped_reference(mat_a, mat_b, offs):
    ref_a = utils.to_reference(mat_a, True)
    ref_b = utils.to_reference(mat_b, True)
    ref_offs = utils.to_reference(offs, False)
    boundaries = [0] + ref_offs.cpu().tolist()
    chunks = [
        ref_a[boundaries[group_idx] : boundaries[group_idx + 1]] @ ref_b[group_idx]
        for group_idx in range(ref_b.shape[0])
    ]
    return torch.cat(chunks, dim=0)


@pytest.mark.grouped_mm
@pytest.mark.group_mm
@pytest.mark.skipif(
    flag_gems.vendor_name != "hygon" and utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("groups,n_size,k_size", GNK_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_grouped_mm(groups, n_size, k_size, dtype):
    mat_a, mat_b, offs = _make_grouped_inputs(groups, n_size, k_size, dtype)
    mat_a_before = utils.to_reference(mat_a.clone(), False)
    mat_b_before = utils.to_reference(mat_b.clone(), False)
    offs_before = utils.to_reference(offs.clone(), False)

    ref_out = _grouped_reference(mat_a, mat_b, offs)
    res_out = flag_gems.group_mm(mat_a, mat_b, offs)

    assert res_out.shape == (mat_a.shape[0], n_size)
    assert res_out.dtype == dtype
    assert res_out.data_ptr() not in {mat_a.data_ptr(), mat_b.data_ptr()}
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=k_size)
    utils.gems_assert_equal(mat_a, mat_a_before)
    utils.gems_assert_equal(mat_b, mat_b_before)
    utils.gems_assert_equal(offs, offs_before)


@pytest.mark.grouped_mm
@pytest.mark.group_mm
@pytest.mark.parametrize("invalid", ["mat_a_rank", "mat_b_rank", "offs_count"])
def test_grouped_mm_rejects_invalid_layout(invalid):
    mat_a, mat_b, offs = _make_grouped_inputs(3, 16, 32, torch.bfloat16)
    if invalid == "mat_a_rank":
        mat_a = mat_a.unsqueeze(0)
    elif invalid == "mat_b_rank":
        mat_b = mat_b.reshape(-1, mat_b.shape[-1])
    else:
        offs = offs[:-1]

    with pytest.raises(AssertionError):
        flag_gems.group_mm(mat_a, mat_b, offs)
