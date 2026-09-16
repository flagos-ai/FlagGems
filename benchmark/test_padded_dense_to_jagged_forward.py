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

import flag_gems

from . import base

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason=(
        "native aten::_padded_dense_to_jagged_forward has no Ascend "
        "PrivateUse1 kernel and falls back to CPU"
    ),
)

PADDED_DENSE_TO_JAGGED_SHAPES = [
    (8, 16, 64),
    (32, 64, 128),
    (128, 128, 64),
    (512, 256, 128),
    (1024, 512, 64),
]
PADDED_DENSE_TO_JAGGED_DTYPES = [
    pytest.param(
        torch.float16,
        marks=[
            pytest.mark.skipif(
                flag_gems.vendor_name == "mthreads",
                reason="the MThreads native FP16 baseline is unavailable",
            ),
            pytest.mark.skipif(
                flag_gems.vendor_name == "iluvatar",
                reason=(
                    "the Iluvatar native FP16 baseline rejects legal large "
                    "core shapes with CUDA error: invalid argument"
                ),
            ),
        ],
    ),
    torch.bfloat16,
    torch.float32,
    pytest.param(
        torch.float64,
        marks=pytest.mark.skipif(
            flag_gems.vendor_name == "iluvatar",
            reason=(
                "the Iluvatar native FP64 baseline returns incorrect all-zero "
                "output for legal inputs"
            ),
        ),
    ),
    torch.int64,
]


class PaddedDenseToJaggedForwardBenchmark(base.Benchmark):
    offset_dtype = torch.int64

    def set_shapes(self, shape_file_path=None):
        self.shapes = PADDED_DENSE_TO_JAGGED_SHAPES

    def get_input_iter(self, cur_dtype):
        for batch_size, max_length, inner_size in self.shapes:
            rng = np.random.default_rng(42)
            lengths = rng.integers(
                max(1, max_length // 4), max_length + 1, size=batch_size
            )
            offsets_values = np.concatenate(([0], np.cumsum(lengths)))
            offsets = torch.tensor(
                offsets_values, device=self.device, dtype=self.offset_dtype
            )
            if cur_dtype == torch.int64:
                dense = torch.randint(
                    -1024,
                    1024,
                    (batch_size, max_length, inner_size),
                    dtype=cur_dtype,
                    device=self.device,
                )
            else:
                dense = torch.randn(
                    (batch_size, max_length, inner_size),
                    dtype=cur_dtype,
                    device=self.device,
                )
            total_L = int(offsets_values[-1])
            yield dense, [offsets], total_L


@pytest.mark.padded_dense_to_jagged_forward
@pytest.mark.parametrize("dtype", PADDED_DENSE_TO_JAGGED_DTYPES)
@pytest.mark.parametrize("offset_dtype", [torch.int32, torch.int64])
def test_padded_dense_to_jagged_forward(dtype, offset_dtype):
    benchmark = PaddedDenseToJaggedForwardBenchmark(
        op_name="padded_dense_to_jagged_forward",
        torch_op=torch.ops.aten._padded_dense_to_jagged_forward,
        gems_op=flag_gems._padded_dense_to_jagged_forward,
        dtypes=[dtype],
    )
    benchmark.offset_dtype = offset_dtype
    benchmark.run()
