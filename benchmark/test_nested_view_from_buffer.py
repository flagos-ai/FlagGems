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

import flag_gems

from . import base, consts


class NestedViewFromBufferBenchmark(base.Benchmark):
    """Benchmark for the zero-copy _nested_view_from_buffer operator."""

    def set_shapes(self, shape_file_path=None):
        # (buffer_size, sizes, strides, offsets) — small / medium / large.
        self.shapes = [
            (6000, [[1000], [2000], [3000]], [[1], [1], [1]], [0, 1000, 3000]),
            (50000, [[500], [1000], [1500]], [[1], [1], [1]], [0, 500, 1500]),
            (200000, [[5000], [10000], [15000]], [[1], [1], [1]], [0, 5000, 15000]),
        ]

    def get_input_iter(self, cur_dtype):
        for buffer_size, sizes, strides, offsets in self.shapes:
            buffer = torch.randn(buffer_size, dtype=cur_dtype, device=self.device)
            # Metadata tensors stay on CPU (CUDA metadata segfaults in the native
            # NestedTensor construction).
            sizes_t = torch.tensor(sizes, dtype=torch.int64, device="cpu")
            strides_t = torch.tensor(strides, dtype=torch.int64, device="cpu")
            offsets_t = torch.tensor(offsets, dtype=torch.int64, device="cpu")
            yield buffer, sizes_t, strides_t, offsets_t

    def get_tflops(self, op, *args, **kwargs):
        # Pure view: no arithmetic work to report.
        return 0.0


@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_view_from_buffer_benchmark(dtype):
    bench = NestedViewFromBufferBenchmark(
        op_name="nested_view_from_buffer",
        torch_op=flag_gems._nested_view_from_buffer,
        dtypes=[dtype],
    )
    bench.run()
