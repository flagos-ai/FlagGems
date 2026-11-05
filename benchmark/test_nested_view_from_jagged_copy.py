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


def _torch_ref(
    values, offsets, dummy, lengths=None, ragged_idx=1, min_seqlen=None, max_seqlen=None
):
    # The native torch path for `_nested_view_from_jagged_copy` clones the flat
    # buffer before building the jagged view; the aten op itself is unavailable
    # on the CUDA backend, so reproduce the semantics via the public jagged API.
    return torch.nested.nested_tensor_from_jagged(
        values.clone(), offsets=offsets, lengths=lengths, jagged_dim=ragged_idx
    )


# Jagged configs: (num_components, inner_dim, total_length)
JAGGED_SHAPES = [
    (4, 8, 64),
    (16, 32, 512),
    (64, 64, 4096),
    (128, 128, 16384),
    (256, 256, 65536),
]


class NestedViewFromJaggedCopyBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = JAGGED_SHAPES

    def get_input_iter(self, cur_dtype):
        for num_components, inner_dim, total in self.shapes:
            values = torch.randn(total, inner_dim, dtype=cur_dtype, device=self.device)
            # Deterministic variable-length split of the packed jagged dimension.
            base_len = total // num_components
            offsets = [0]
            for i in range(num_components):
                length = base_len + (1 if i < (total % num_components) else 0)
                offsets.append(offsets[-1] + length)
            offsets = torch.tensor(offsets, dtype=torch.int64, device=self.device)
            dummy = torch.randn(
                num_components, inner_dim, dtype=cur_dtype, device=self.device
            )
            yield values, offsets, dummy

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_jagged_copy
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test__nested_view_from_jagged_copy(dtype):
    bench = NestedViewFromJaggedCopyBenchmark(
        op_name="nested_view_from_jagged_copy",
        torch_op=_torch_ref,
        gems_op=flag_gems._nested_view_from_jagged_copy,
        dtypes=[dtype],
    )
    bench.run()
