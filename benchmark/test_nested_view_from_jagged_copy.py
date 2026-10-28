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

    def get_case_iter(self, dtype):
        for ordinal, shape in enumerate(self.shapes):
            num_components, inner_dim, total = shape
            yield self._case_from_plan(
                dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"input": [total, inner_dim]},
                    builder_args=(shape,),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        (shape,) = plan.builder_args
        num_components, inner_dim, total = shape
        cur_dtype = case.dtype
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
        return values, offsets, dummy

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_jagged_copy
@pytest.mark.underscore_nested_view_from_jagged_copy
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test__nested_view_from_jagged_copy(dtype):
    bench = NestedViewFromJaggedCopyBenchmark(
        op_name="_nested_view_from_jagged_copy",
        torch_op=_torch_ref,
        dtypes=[dtype],
    )
    bench.run()
