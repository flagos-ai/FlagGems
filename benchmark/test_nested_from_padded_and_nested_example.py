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


class NestedFromPaddedAndNestedExampleBenchmark(base.Benchmark):
    """Benchmark for the _nested_from_padded_and_nested_example operator."""

    def set_shapes(self, shape_file_path=None):
        # (B, D_pad, D, lengths) configurations covering small/medium/large cases,
        # where only the first dimension is ragged.
        self.shapes = [
            (4, 16, 1, [3, 9, 5, 13]),
            (4, 16, 8, [3, 9, 5, 13]),
            (8, 64, 32, [13, 29, 51, 7, 33, 47, 21, 55]),
            (
                16,
                128,
                64,
                [31, 97, 63, 15, 111, 7, 79, 45, 23, 89, 57, 37, 103, 67, 13, 121],
            ),
        ]
        # Trailing-ragged: (padded_shape, nested_sizes) pairs that exercise the
        # general kernel path, where trailing dimensions are also ragged.
        self.ragged_shapes = [
            ((2, 8, 4), [[3, 4], [5, 2]]),
            ((2, 8, 4, 3), [[3, 4, 3], [5, 2, 1]]),
        ]
        self.shape_desc = "B, D_pad, D, lengths (+ trailing-ragged sizes)"

    def _make_inputs(self, padded_shape, sizes, cur_dtype):
        padded = torch.randn(padded_shape, dtype=cur_dtype, device=self.device)
        data = torch.randn(padded_shape, dtype=cur_dtype, device=self.device)
        if not isinstance(sizes, torch.Tensor):
            sizes = torch.tensor(sizes, dtype=torch.int64)
        nt_example = torch.ops.aten._nested_from_padded(data, sizes)
        return padded, nt_example

    def get_input_iter(self, cur_dtype):
        for B, D_pad, D, lengths in self.shapes:
            lengths_t = torch.tensor(lengths, dtype=torch.int64)
            if D == 1:
                sizes = lengths_t.reshape(-1, 1)
            else:
                sizes = torch.stack(
                    [lengths_t, torch.full((B,), D, dtype=torch.int64)], dim=1
                )
            padded_shape = (B, D_pad) if D == 1 else (B, D_pad, D)
            yield self._make_inputs(padded_shape, sizes, cur_dtype)

        for padded_shape, sizes in self.ragged_shapes:
            yield self._make_inputs(padded_shape, sizes, cur_dtype)

    def get_tflops(self, op, *args, **kwargs):
        return 0.0

    def record_shapes(self, *args, **kwargs):
        # args == (padded, nt_example). Legacy nested tensor sizes contain
        # SymInts that break the default JSON serializer, so emit a plain
        # string description instead.
        padded, _ = args
        return f"padded={tuple(padded.shape)}"


@pytest.mark.nested_from_padded_and_nested_example
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_from_padded_and_nested_example(dtype):
    bench = NestedFromPaddedAndNestedExampleBenchmark(
        op_name="nested_from_padded_and_nested_example",
        torch_op=torch.ops.aten._nested_from_padded_and_nested_example,
        dtypes=[dtype],
    )
    bench.set_gems(flag_gems._nested_from_padded_and_nested_example)
    bench.run()
