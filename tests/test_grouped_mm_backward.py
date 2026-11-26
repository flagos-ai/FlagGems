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

GNK_SHAPES = [(2, 128, 32), (4, 256, 64), (8, 512, 128)]
# (M, K, N, G) small shapes for the functional entry point
MKG_SHAPES = [
    (64, 128, 32, 2),
    (96, 128, 64, 4),
    (512, 256, 128, 8),
    (256, 2048, 512, 2),
]


def _make_inputs(m, k, n, groups, device, dtype=torch.bfloat16):
    mat1 = torch.randn(m, k, device=device, dtype=dtype)
    mat2 = torch.randn(groups, k, n, device=device, dtype=dtype)
    grad = torch.randn(m, n, device=device, dtype=dtype)
    # non-empty groups: split m into `groups` ascending cumulative ends
    base = m // groups
    offs = torch.arange(1, groups + 1, device=device, dtype=torch.int32) * base
    offs[-1] = m
    return grad, mat1, mat2, offs


def _reference(grad, mat1, mat2, offs):
    """ATen row-major fast paths of the _grouped_mm backward helpers."""
    if utils.TO_CPU:
        ref_grad_mat1 = torch.ops.aten._grouped_mm(
            grad.cpu(), mat2.cpu().transpose(-2, -1), offs.cpu()
        )
        ref_grad_mat2 = torch.ops.aten._grouped_mm(
            grad.cpu().transpose(-2, -1), mat1.cpu(), offs.cpu()
        ).transpose(-2, -1)
    else:
        ref_grad_mat1 = torch.ops.aten._grouped_mm(grad, mat2.transpose(-2, -1), offs)
        ref_grad_mat2 = torch.ops.aten._grouped_mm(
            grad.transpose(-2, -1), mat1, offs
        ).transpose(-2, -1)
    return ref_grad_mat1, ref_grad_mat2


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("m, k, n, groups", MKG_SHAPES)
def test_grouped_mm_backward_functional(m, k, n, groups):
    """The explicit ``grouped_mm_backward(grad, mat1, mat2, offs)`` entry."""
    grad, mat1, mat2, offs = _make_inputs(m, k, n, groups, flag_gems.device)

    ref_grad_mat1, ref_grad_mat2 = _reference(grad, mat1, mat2, offs)
    res_grad_mat1, res_grad_mat2 = flag_gems.grouped_mm_backward(grad, mat1, mat2, offs)

    utils.gems_assert_close(res_grad_mat1, ref_grad_mat1, torch.bfloat16)
    utils.gems_assert_close(res_grad_mat2, ref_grad_mat2, torch.bfloat16)


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
@pytest.mark.parametrize("groups, n, k", GNK_SHAPES)
def test_grouped_mm_backward_autograd(groups, n, k):
    """The gradient flows through the FlagGems bridge for ``group_mm``-like
    forwards driven by native ``_grouped_mm`` outputs."""
    dtype = torch.bfloat16
    m = 0
    m_list = []
    for _ in range(groups):
        m_g = max(1, 128 // groups) * (_ % 3 + 1)
        m_list.append(m_g)
        m += m_g
    mat1 = torch.randn(m, k, device=flag_gems.device, dtype=dtype)
    mat2 = torch.randn(groups, k, n, device=flag_gems.device, dtype=dtype)
    offs = torch.tensor(
        [sum(m_list[: i + 1]) for i in range(groups)],
        dtype=torch.int32,
        device=flag_gems.device,
    )
    grad = torch.randn(m, n, device=flag_gems.device, dtype=dtype)

    ref_grad_mat1, ref_grad_mat2 = _reference(grad, mat1, mat2, offs)
    res_grad_mat1, res_grad_mat2 = flag_gems.grouped_mm_backward(grad, mat1, mat2, offs)

    utils.gems_assert_close(res_grad_mat1, ref_grad_mat1, dtype)
    utils.gems_assert_close(res_grad_mat2, ref_grad_mat2, dtype)


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm_backward_matches_autograd_grad():
    """grad from native autograd equals the two formulas bit-exact."""
    m, k, n, groups = 64, 128, 64, 2
    grad, mat1, mat2, offs = _make_inputs(m, k, n, groups, flag_gems.device)
    if utils.TO_CPU:
        pytest.skip("native autograd reference requires CUDA _grouped_mm")

    mat1_rg = mat1.clone().requires_grad_(True)
    mat2_rg = mat2.clone().requires_grad_(True)
    out = torch.ops.aten._grouped_mm(mat1_rg, mat2_rg, offs)
    ga, gb = torch.autograd.grad(out, [mat1_rg, mat2_rg], grad)

    res_ga, res_gb = flag_gems.grouped_mm_backward(grad, mat1, mat2, offs)
    utils.gems_assert_equal(res_ga, ga)
    utils.gems_assert_equal(res_gb, gb)


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm_backward_non_contiguous():
    """Non-contiguous mat1/grad inputs still produce the right gradients."""
    m, k, n, groups = 64, 128, 64, 2
    grad, mat1, mat2, offs = _make_inputs(m, k, n, groups, flag_gems.device)

    mat1_t = mat1.t().contiguous().t()  # (M, K) column-major
    assert not mat1_t.is_contiguous()

    ref_grad_mat1, ref_grad_mat2 = _reference(grad, mat1, mat2, offs)
    res_grad_mat1, res_grad_mat2 = flag_gems.grouped_mm_backward(
        grad, mat1_t, mat2, offs
    )
    utils.gems_assert_close(res_grad_mat1, ref_grad_mat1, torch.bfloat16)
    utils.gems_assert_close(res_grad_mat2, ref_grad_mat2, torch.bfloat16)


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm_backward_3d_uniform():
    """3-D mat1 without offs: uniform groups."""
    g, m, k, n = 2, 32, 128, 64
    mat1_3d = torch.randn(g, m, k, device=flag_gems.device, dtype=torch.bfloat16)
    mat2 = torch.randn(g, k, n, device=flag_gems.device, dtype=torch.bfloat16)
    grad3 = torch.randn(g, m, n, device=flag_gems.device, dtype=torch.bfloat16)

    # reference on the flattened 2-D view with uniform offs
    offs = torch.arange(1, g + 1, device=flag_gems.device, dtype=torch.int32) * m
    ref_ga, ref_gb = _reference(
        grad3.reshape(g * m, n), mat1_3d.reshape(g * m, k), mat2, offs
    )
    ref_ga = ref_ga.view(g, m, k)

    res_ga, res_gb = flag_gems.grouped_mm_backward(grad3, mat1_3d, mat2)
    utils.gems_assert_close(res_ga, ref_ga, torch.bfloat16)
    utils.gems_assert_close(res_gb, ref_gb, torch.bfloat16)
