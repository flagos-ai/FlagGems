# Copyright 2026, The FlagOS Contributors.
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

# _pack_padded_sequence benchmark
# (time_steps, batch_size, feat_size): RNN-style padded batches of growing size
PACK_PADDED_SEQUENCE_SHAPES = [
    (16, 8, 64),
    (32, 16, 128),
    (64, 32, 256),
    (128, 64, 256),
    (256, 128, 512),
]


class PackPaddedSequenceBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = PACK_PADDED_SEQUENCE_SHAPES

    def get_input_iter(self, cur_dtype):
        for time_steps, batch_size, feat_size in self.shapes:
            inp = torch.randn(
                time_steps, batch_size, feat_size, dtype=cur_dtype, device=self.device
            )
            # lengths must be a 1-D CPU int64 tensor sorted in descending order.
            lengths = torch.linspace(
                time_steps, 1, steps=batch_size, dtype=torch.int64
            ).clamp(min=1)
            lengths, _ = torch.sort(lengths, descending=True)
            yield inp, lengths, False


@pytest.mark.pack_padded_sequence
def test_pack_padded_sequence():
    bench = PackPaddedSequenceBenchmark(
        op_name="pack_padded_sequence",
        torch_op=torch.ops.aten._pack_padded_sequence,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
