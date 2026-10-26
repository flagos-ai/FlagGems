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

from . import base, consts

# Pad packed sequence benchmark. Each pair is (batch_size, max_length); the
# feature width is fixed at 128. Shapes span small to large batch/time extents
# to exercise both padding-heavy and copy-heavy scatters.
PAD_PACKED_SEQUENCE_SHAPES = [
    (8, 8),
    (16, 16),
    (32, 32),
    (64, 64),
    (128, 64),
    (256, 128),
    (512, 256),
]


class PadPackedSequenceBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PAD_PACKED_SEQUENCE_SHAPES

    def get_case_iter(self, cur_dtype):
        for ordinal, (batch_size, max_length) in enumerate(self.shapes):
            yield self._case_from_plan(
                cur_dtype,
                ordinal,
                base.BenchmarkCasePlan(
                    shape={"batch_size": batch_size, "max_length": max_length},
                    params={"batch_first": False, "padding_value": 0.0},
                    builder_args=(batch_size, max_length),
                ),
            )

    def build_inputs(self, case):
        plan = case.builder_args[0]
        batch_size, max_length = plan.builder_args
        cur_dtype = case.dtype

        np.random.seed(42)
        lengths = np.random.randint(1, max_length + 1, size=batch_size).tolist()
        lengths = sorted(lengths, reverse=True)

        max_len = max(lengths)
        batch_sizes = torch.tensor(
            [int(sum(1 for length in lengths if length > t)) for t in range(max_len)],
            dtype=torch.int64,
        )
        if base.vendor_name == "ascend" and (
            torch.any(batch_sizes != batch_sizes[0]).item()
            or max_length != batch_sizes.numel()
        ):
            pytest.skip(
                "Ascend native _pad_packed_sequence requires constant batch sizes "
                "and total_length equal to the packed time dimension"
            )
        total = int(batch_sizes.sum().item())
        # Fixed feature width of 128 keeps the packed tensor 2D and memory
        # traffic representative of typical RNN hidden sizes.
        data = torch.randn(total, 128, dtype=cur_dtype, device=self.device)

        return data, batch_sizes, False, 0.0, max_length


@pytest.mark.pad_packed_sequence
@pytest.mark.underscore_pad_packed_sequence
def test_pad_packed_sequence():
    bench = PadPackedSequenceBenchmark(
        op_name="_pad_packed_sequence",
        torch_op=torch.ops.aten._pad_packed_sequence,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
