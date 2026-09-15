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
from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts, utils

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None


class GroupmmBenchmark(base.BlasBenchmark):
    def get_input_iter(self, dtype) -> Generator:
        for groups, n, k in self.shapes:
            yield from self.input_fn(groups, n, k, dtype, self.device)

    def set_more_shapes(self):
        return []

    def get_tflops(self, op, *args, **kwargs):
        groups, N, K = args[1].shape
        size_per_group = torch.diff(
            args[2], prepend=torch.zeros(1, device=args[2].device, dtype=torch.int32)
        )
        total_flops = 0
        for i in range(groups):
            total_flops += size_per_group[i].item() * N * K * 2
        return total_flops


def _input_fn(groups, N, K, cur_dtype, device):
    assert cur_dtype == torch.bfloat16

    group_A_list = []
    group_B_list = []
    A_offs = 0
    B_offs = 0
    M_list = []
    for i in range(groups):
        M_g = random.randint(1, 16384)
        N_g = N
        K_g = K
        A_g = torch.rand([M_g, K_g], device=device, dtype=cur_dtype)
        B_g = torch.rand([K_g, N_g], device=device, dtype=cur_dtype)
        group_A_list.append(A_g)
        group_B_list.append(B_g)
        M_list.append(M_g)
        A_offs += M_g * K_g
        B_offs += K_g * N_g

    mat_a = torch.cat([x for x in group_A_list], dim=0)
    mat_b = torch.stack([x for x in group_B_list], dim=0)
    offs = torch.tensor(
        [sum(M_list[: i + 1]) for i in range(groups)], dtype=torch.int32, device=device
    )

    yield mat_a, mat_b, offs


@pytest.mark.grouped_mm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm(monkeypatch):
    bench = GroupmmBenchmark(
        op_name="grouped_mm",
        input_fn=_input_fn,
        torch_op=torch._grouped_mm,
        gems_op=flag_gems.group_mm,
        dtypes=[torch.bfloat16],
    )

    bench.run()


# ---------------------------------------------------------------------------
# Ascend specialized grouped_matmul perf (torch_npu.npu_grouped_matmul
# baseline): same jagged grouped GEMM problem, fixed M shapes with random
# per-group boundaries and both group_list semantics.
# ---------------------------------------------------------------------------

ascend_only = pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="the optimized Grouped MatMul implementation targets Ascend",
)

GROUPED_MATMUL_SHAPES = (
    (1024, 2048, 4096, 32),
    (1024, 4096, 1024, 32),
    (2560, 2048, 4096, 32),
    (2560, 4096, 1024, 32),
    (3339, 4096, 1024, 32),
    (7354, 4096, 1024, 32),
    (7354, 2048, 4096, 32),
    (4096, 4096, 2048, 32),
)


def _make_group_list(m, groups, group_list_type, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    boundaries = torch.floor(torch.rand(groups - 1, generator=generator) * m)
    cumulative = torch.cat(
        (
            torch.sort(boundaries)[0].to(torch.int64),
            torch.tensor([m], dtype=torch.int64),
        )
    )
    if group_list_type == 0:
        return cumulative.to(device=flag_gems.device)
    counts = cumulative.clone()
    counts[1:] -= cumulative[:-1]
    return counts.to(device=flag_gems.device)


def _official(x, weight, group_list, group_list_type):
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight],
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=group_list_type,
    )[0]


class GroupedMatmulBenchmark(base.Benchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]
    DEFAULT_SHAPE_DESC = "M, N, K, groups, group_list_type"
    DEFAULT_SHAPES = tuple(
        (*shape, group_list_type)
        for shape in GROUPED_MATMUL_SHAPES
        for group_list_type in (0, 1)
    )

    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has a generic `Benchmark:` fallback that would be
        # matched via the class MRO and override DEFAULT_SHAPES with
        # unrelated shapes.
        self.shapes = self.DEFAULT_SHAPES

    def get_input_iter(self, dtype) -> Generator:
        for case_index, (m, n, k, groups, group_list_type) in enumerate(self.shapes):
            torch.manual_seed(1000 + case_index // 2)
            x = torch.rand((m, k), device=self.device, dtype=dtype)
            weight = torch.rand((groups, k, n), device=self.device, dtype=dtype)
            group_list = _make_group_list(
                m,
                groups,
                group_list_type,
                2026 + case_index // 2,
            )
            yield x, weight, group_list, group_list_type

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        weight = args[1]
        m, k = x.shape
        n = weight.shape[2]
        return 2 * m * n * k


@pytest.mark.grouped_matmul
@ascend_only
def test_grouped_matmul_perf():
    benchmark = GroupedMatmulBenchmark(
        op_name="grouped_matmul",
        torch_op=_official,
        gems_op=flag_gems.grouped_matmul,
        dtypes=[torch.float16],
    )
    benchmark.run()
