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

import math

import pytest
import torch

import flag_gems

from . import base, consts

MUL_SHAPES = [
    ((64, 64), (64, 64)),
    ((2048, 2048), (2048, 2048)),
    ((4096, 4096), (4096, 4096)),
    ((64, 512, 512), (64, 512, 512)),
    ((1024, 1024, 1024), (1024, 1024, 1024)),
    ((128, 1), (128, 2048)),
    ((8192, 1), (1, 18)),
    ((18,), ()),
    ((512, 2048), ()),
]


class MulBenchmark(base.BinaryPointwiseBenchmark):
    DEFAULT_SHAPE_DESC = "shape"

    def set_shapes(self, shape_file_path=None):
        self.shapes = MUL_SHAPES

    def get_input_iter(self, dtype):
        for index, (lhs_shape, rhs_shape) in enumerate(self.shapes):
            lhs = base.generate_tensor_input(lhs_shape, dtype, self.device)
            rhs = (
                base.generate_tensor_input(rhs_shape, dtype, self.device)
                if rhs_shape
                else 0.5
            )
            yield lhs, rhs
            if index == 0:
                # A single shared-output case covers the common fallback kernel.
                rhs = torch.ones_like(lhs)
                yield lhs, rhs, {"out": lhs}

    def get_tflops(self, op, *args, **kwargs):
        rhs_shape = args[1].shape if isinstance(args[1], torch.Tensor) else ()
        return math.prod(torch.broadcast_shapes(args[0].shape, rhs_shape))


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.mul
def test_mul():
    bench = MulBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
        # dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
    bench.set_gems(flag_gems.mul)
    bench.run()


@pytest.mark.mul_
def test_mul_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul_",
        torch_op=lambda a, b: a.mul_(b),
        dtypes=consts.FLOAT_DTYPES,
        is_inplace=True,
    )
    bench.run()
