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


class SliceBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        SLICE_SHAPES = (
            (128, 256),
            (1024, 1024),
            (512, 1024, 512),
            (16, 8192, 4096),
            (8, 4096, 11008),
            (4, 32, 4096, 128),
            (32, 256, 256, 128),
        )

        self.shapes = SLICE_SHAPES
        return None


def _get_gbps(args, latency):
    inp, dim, start, end, step = args

    bytes_per_element = inp.element_size()

    output_numel = 1
    for i, s in enumerate(inp.shape):
        if i == dim:
            output_numel *= (end - start + step - 1) // step
        else:
            output_numel *= s

    total_bytes = (inp.numel() + output_numel) * bytes_per_element

    return total_bytes / latency / 1e9


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1

    start = 0
    end = shape[dim]
    step = 2

    size = shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    inp = torch.randn(
        shape,
        dtype=dtype,
        device=device,
    )

    yield inp, dim, start, end, step


@pytest.mark.slice
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_slice():
    bench = SliceBenchmark(
        op_name="slice",
        torch_op=torch.ops.aten.slice.Tensor,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
