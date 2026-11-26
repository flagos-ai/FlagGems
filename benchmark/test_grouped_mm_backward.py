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

from typing import Generator

import pytest
import torch

import flag_gems

from . import base, utils


class GroupedMmBackwardBenchmark(base.GenericBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for groups, n, k in self.shapes:
            yield from self.input_fn(groups, n, k, dtype, self.device)

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        grad, mat1, mat2, offs = args[:4]
        n_dim = grad.shape[1]
        size_per_group = torch.diff(
            offs, prepend=torch.zeros(1, device=offs.device, dtype=torch.int32)
        )
        # dmat1: m_i x k x n x 2 ; dmat2: m_i x k x n x 2
        total_flops = 0
        for i in range(mat2.shape[0]):
            total_flops += size_per_group[i].item() * mat1.shape[1] * n_dim * 4
        return total_flops


def _input_fn(groups, N, K, cur_dtype, device):
    assert cur_dtype == torch.bfloat16
    m_list = []
    for i in range(groups):
        m_g = max(1, 1024 // groups) * (i % 3 + 1)
        m_list.append(m_g)
    m = sum(m_list)
    mat1 = torch.randn(m, K, device=device, dtype=cur_dtype)
    mat2 = torch.randn(groups, K, N, device=device, dtype=cur_dtype)
    grad = torch.randn(m, N, device=device, dtype=cur_dtype)
    offs = torch.tensor(
        [sum(m_list[: i + 1]) for i in range(groups)],
        dtype=torch.int32,
        device=device,
    )
    yield grad, mat1, mat2, offs


@pytest.mark.grouped_mm_backward
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm_backward(monkeypatch):
    bench = GroupedMmBackwardBenchmark(
        op_name="grouped_mm_backward",
        input_fn=_input_fn,
        torch_op=_aten_grouped_mm_backward,
        gems_op=flag_gems.grouped_mm_backward,
        dtypes=[torch.bfloat16],
    )

    bench.run()


def _aten_grouped_mm_backward(grad, mat1, mat2, offs):
    """ATen row-major fast paths of the _grouped_mm backward helpers."""
    grad_mat1 = torch.ops.aten._grouped_mm(grad, mat2.transpose(-2, -1), offs)
    grad_mat2 = torch.ops.aten._grouped_mm(
        grad.transpose(-2, -1), mat1, offs
    ).transpose(-2, -1)
    return grad_mat1, grad_mat2
