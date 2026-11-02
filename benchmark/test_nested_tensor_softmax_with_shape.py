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

# (B, H, T_max, seq_lens)
NESTED_SOFTMAX_SHAPES = [
    (2, 2, 16, [8, 16]),
    (4, 2, 32, [16, 24, 32, 32]),
    (8, 4, 64, [16, 32, 48, 64, 16, 32, 48, 64]),
]


def _torch_ref(self_nt, query_nt):
    """Reference masked softmax implemented with native torch ops."""
    attn = self_nt.to_padded_tensor(0.0)
    offsets = query_nt.offsets()
    seq_lens = offsets[1:] - offsets[:-1]
    B, T_max, H, _ = attn.shape
    out = torch.zeros_like(attn)
    for b in range(B):
        T = int(seq_lens[b])
        if T <= 0:
            continue
        out[b, :T, :, :T] = torch.softmax(attn[b, :T, :, :T], dim=-1)
    return out


class NestedTensorSoftmaxWithShapeBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_SOFTMAX_SHAPES

    def get_input_iter(self, cur_dtype):
        for B, H, T_max, seq_lens in self.shapes:
            self_components = [
                torch.randn(seq_lens[b], H, T_max, dtype=cur_dtype, device=self.device)
                for b in range(B)
            ]
            self_nt = torch.nested.nested_tensor(
                self_components, layout=torch.jagged, device=self.device
            )
            query_components = [
                torch.randn(seq_lens[b], 4, dtype=cur_dtype, device=self.device)
                for b in range(B)
            ]
            query_nt = torch.nested.nested_tensor(
                query_components, layout=torch.jagged, device=self.device
            )
            yield self_nt, query_nt

    def get_tflops(self, op, *args, **kwargs):
        return 0.0

    def record_shapes(self, *args, **kwargs):
        # Nested tensors report a ragged (SymInt) dim; record only concrete dims
        # so the result can be JSON-serialized.
        self_nt = args[0]
        return [(int(self_nt.size(0)), int(self_nt.size(2)), int(self_nt.size(3)))]


@pytest.mark.nested_tensor_softmax_with_shape
@pytest.mark.nested_tensor_softmax_with_shape
@pytest.mark.nested_tensor_softmax_with_shape
def test_nested_tensor_softmax_with_shape():
    bench = NestedTensorSoftmaxWithShapeBenchmark(
        op_name="nested_tensor_softmax_with_shape",
        torch_op=_torch_ref,
        dtypes=consts.FLOAT_DTYPES,
        gems_op=flag_gems._nested_tensor_softmax_with_shape,
    )
    bench.run()
