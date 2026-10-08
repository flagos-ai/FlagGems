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

import numpy as np
import pytest
import torch
from torch.nested._internal.nested_tensor import _nt_view_dummy

import flag_gems

from . import base, consts

# (B, T, D) padded shapes spanning small-to-large batch/length/feature so the
# un-padding kernel is exercised across a range of inner-dim widths.
NESTED_FROM_PADDED_SHAPES = [
    (8, 16, 16),
    (32, 32, 64),
    (128, 64, 256),
    (512, 128, 64),
]

# `_nt_view_dummy()` is a device-agnostic (meta) nested tensor used purely as a
# layout hint by both the ATen op and the FlagGems implementation.
_DUMMY = _nt_view_dummy()


def _native_op(padded, offsets, ragged_idx):
    return torch.ops.aten._nested_from_padded_tensor.default(
        padded, offsets, _DUMMY, ragged_idx, None, None, None
    )


def _gems_op(padded, offsets, ragged_idx):
    return flag_gems._nested_from_padded_tensor(
        padded, offsets, _DUMMY, ragged_idx, None, None, None
    )


class NestedFromPaddedTensorBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_FROM_PADDED_SHAPES

    def get_input_iter(self, cur_dtype):
        for B, T, D in self.shapes:
            np.random.seed(42)
            lengths = np.random.randint(1, T + 1, size=B)
            offsets = torch.tensor(
                [0] + np.cumsum(lengths).tolist(),
                dtype=torch.int32,
                device=self.device,
            )
            padded = torch.randn(B, T, D, dtype=cur_dtype, device=self.device)
            yield padded, offsets, 1


@pytest.mark.nested_from_padded_tensor
def test_nested_from_padded_tensor():
    bench = NestedFromPaddedTensorBenchmark(
        op_name="nested_from_padded_tensor",
        torch_op=_native_op,
        gems_op=_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
