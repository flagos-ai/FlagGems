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
from flag_gems.utils import shape_utils

from . import base, consts, utils


class TensorSelectBenchmark(base.GenericBenchmark2DOnly):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        # Speed Up Benchmark Test, Big Shape Will Cause Timeout
        if flag_gems.vendor_name == "kunlunxin":
            return []

        shapes = super().set_more_shapes()
        shapes = [
            shape
            for shape in shapes
            if len(shape) == 2 and shape[0] > 16 and shape[1] > 16
        ]
        return shapes


def narrow_copy_input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    dim = 0
    start = shape[dim] // 4
    length = shape[dim] // 2
    yield inp, dim, start, length


def narrow_copy_gbps(bench_fn_args, latency):
    inp, dim, start, length = bench_fn_args
    # Input is full tensor, output is a slice
    io_amount = shape_utils.size_in_bytes(inp)
    # We read the full input and write the output
    return io_amount * 1e-9 / (latency * 1e-3)


@pytest.mark.narrow_copy
def test_narrow_copy_perf():
    bench = TensorSelectBenchmark(
        op_name="narrow_copy",
        torch_op=torch.narrow_copy,
        input_fn=narrow_copy_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=narrow_copy_gbps,
    )
    bench.run()
