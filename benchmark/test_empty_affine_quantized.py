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
"""Benchmark for ``aten::_empty_affine_quantized``.

The operator is a pure factory: the timed work is the device allocation and the
construction of the per-tensor quantizer.  There is no input tensor and no
arithmetic, so the payload is the requested size plus the fixed quantizer
metadata (scale, zero point).  The ATen factory is the baseline; the FlagGems
candidate is reached by calling the same ATen entry point under
``use_gems()``.
"""

import pytest
import torch

from . import base

# Allocation-cost sweep: the two small rows are launch/host-bound, the larger
# rows make the device allocation dominant.
SIZES = [
    (256, 64),
    (1024, 1024),
    (20, 320, 15),
    (16, 128, 64, 60),
]


def _ref(size, *, dtype, device, scale, zero_point):
    return torch.ops.aten._empty_affine_quantized.default(
        size,
        dtype=dtype,
        device=device,
        scale=scale,
        zero_point=zero_point,
        memory_format=torch.contiguous_format,
    )


class EmptyAffineQuantizedBenchmark(base.Benchmark):
    """Per-tensor quantized allocation; no input tensor, no value checks."""

    DEFAULT_SHAPE_DESC = "size"

    def set_shapes(self, shape_file_path=None):
        # core_shapes.yaml has no entry for this factory; the curated list above
        # keeps the allocation sizes explicit.
        del shape_file_path
        self.shapes = list(SIZES)

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield list(shape), {
                "dtype": cur_dtype,
                "device": self.device,
                "scale": 0.25,
                "zero_point": 0,
            }


@pytest.mark.empty_affine_quantized
def test_empty_affine_quantized():
    bench = EmptyAffineQuantizedBenchmark(
        op_name="empty_affine_quantized",
        torch_op=_ref,
        dtypes=[torch.quint8, torch.qint8, torch.qint32],
    )
    bench.run()
