# Copyright 2026 XCoreSigma Contributors
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

from . import base, consts

try:
    import torch_npu  # noqa: F401
except ImportError:
    torch_npu = None

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
        dtypes=[torch.float16, torch.bfloat16],
    )
    benchmark.run()
