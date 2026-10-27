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

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

GROUPED_MM_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


class GroupedMMBenchmark(OperatorBenchmark):
    DEFAULT_METRICS = consts.DEFAULT_METRICS[:] + ["tflops"]

    def set_shapes(self, shape_file_path=None):
        self.shapes = GROUPED_MM_SHAPES
        self.shape_desc = "groups, N, K"

    def get_tflops(self, op, *args, **kwargs):
        del op, kwargs
        mat_b, offs = args[1], args[2]
        groups, k_size, n_size = mat_b.shape
        boundaries = [0] + offs.cpu().tolist()
        return sum(
            (boundaries[idx + 1] - boundaries[idx]) * n_size * k_size * 2
            for idx in range(groups)
        )


def _case_fn(shape, dtype):
    del dtype
    groups, n_size, k_size = shape
    group_sizes = tuple(16 + (group_idx * 13) % 48 for group_idx in range(groups))
    yield base.BenchmarkCasePlan(
        shape={
            "mat_a": [sum(group_sizes), k_size],
            "mat_b": [groups, k_size, n_size],
            "offs": [groups],
        },
        params={"group_sizes": list(group_sizes)},
        builder_args=(groups, n_size, k_size, group_sizes),
    )


def _build_inputs_fn(plan, dtype, device):
    groups, n_size, k_size, group_sizes = plan.builder_args
    torch.manual_seed(0)
    mat_a = torch.rand((sum(group_sizes), k_size), dtype=dtype, device=device)
    mat_b = torch.rand((groups, k_size, n_size), dtype=dtype, device=device)
    offs = torch.tensor(
        [sum(group_sizes[: idx + 1]) for idx in range(groups)],
        dtype=torch.int32,
        device=device,
    )
    return mat_a, mat_b, offs


@pytest.mark.grouped_mm
@pytest.mark.group_mm
@pytest.mark.skipif(
    utils.SkipVersion("torch", "<2.8"),
    reason="torch._grouped_mm requires PyTorch >= 2.8.0.",
)
def test_grouped_mm():
    GroupedMMBenchmark(
        op_name="group_mm",
        torch_op=torch._grouped_mm,
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        dtypes=[torch.bfloat16],
    ).run()
