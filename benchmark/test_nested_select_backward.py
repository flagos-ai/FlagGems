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

from . import base, consts


class NestedSelectBackwardBenchmark(base.Benchmark):
    """Benchmark for the _nested_select_backward operator."""

    def set_shapes(self, shape_file_path=None):
        # (component sizes along the ragged dim, dense dim) configurations.
        self.shapes = [
            ([5, 7, 4], 3),
            ([16, 8, 12, 20], 8),
            ([64, 128, 256], 16),
            ([128, 64, 192, 256], 32),
        ]
        self.shape_desc = "component_sizes, dense_dim"

    def get_input_iter(self, cur_dtype):
        for sizes, dense_dim in self.shapes:
            comps = [
                torch.randn(s, dense_dim, dtype=cur_dtype, device=self.device)
                for s in sizes
            ]
            self_nt = torch.nested.nested_tensor(
                comps, layout=torch.jagged, device=self.device
            )
            index = 1
            grad = torch.randn(
                sizes[index], dense_dim, dtype=cur_dtype, device=self.device
            )
            yield grad, self_nt, 0, index

    def get_tflops(self, op, *args, **kwargs):
        return 0.0

    def record_shapes(self, *args, **kwargs):
        # args == (grad, nested_self, dim, index). Nested tensor sizes contain
        # SymInts that break the default JSON serializer, so emit a plain
        # string description instead.
        grad, self_nt, dim, index = args
        comp_sizes = [int(t.shape[0]) for t in self_nt.unbind()]
        return f"components={comp_sizes} dense_dim={grad.shape[-1]} index={index}"


@pytest.mark.nested_select_backward
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_select_backward(dtype):
    bench = NestedSelectBackwardBenchmark(
        op_name="nested_select_backward",
        torch_op=torch.ops.aten._nested_select_backward,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems._nested_select_backward)
    bench.run()
