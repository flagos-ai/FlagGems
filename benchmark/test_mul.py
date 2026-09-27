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


ASCEND_MUL_ROW_SIZES = [
    1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104,
    112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208,
    216, 224, 232, 240, 248, 256, 272, 288, 304, 320, 336, 352, 368,
    384, 400, 416, 432, 448, 464, 480, 496, 512, 1035, 1036, 2048,
    4106, 4107, 4108, 4109, 4434, 4435, 6207, 9331, 10342, 10343,
    10441, 13421, 13422, 14554, 16384,
]

ASCEND_MUL_SHAPES = [
    ("broadcast", (m, 1), (m, 2048)) for m in ASCEND_MUL_ROW_SIZES
] + [
    ("broadcast", (1048576, 1), (1, 32)),
    ("broadcast", (8192, 1), (1, 18)),
    ("scalar", (18,), None),
    ("scalar", (32,), None),
    ("scalar", (1, 2048), None),
    ("scalar", (2, 2048), None),
    ("scalar", (4, 2048), None),
    ("scalar", (8, 2048), None),
    ("scalar", (16, 2048), None),
    ("scalar", (32, 2048), None),
    ("scalar", (64, 2048), None),
    ("scalar", (128, 2048), None),
    ("scalar", (256, 2048), None),
    ("scalar", (512, 2048), None),
]


class AscendMulBenchmark(base.BinaryPointwiseBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = ASCEND_MUL_SHAPES
        self.shape_desc = "kind, lhs_shape, rhs_shape"

    def get_input_iter(self, dtype):
        for kind, lhs_shape, rhs_shape in self.shapes:
            lhs = base.generate_tensor_input(lhs_shape, dtype, self.device)
            if kind == "scalar":
                yield lhs, 0.5
            else:
                rhs = base.generate_tensor_input(rhs_shape, dtype, self.device)
                yield lhs, rhs


@pytest.mark.mul
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend-specific real-shape mul benchmark",
)
def test_mul_ascend_real_shapes():
    bench = AscendMulBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.set_gems(flag_gems.mul)
    bench.run()


# TODO(0x45f): Fix OOM when dtypes includes COMPLEX_DTYPES is included (Issue #2693).
@pytest.mark.mul
def test_mul():
    bench = base.BinaryPointwiseBenchmark(
        op_name="mul",
        torch_op=torch.mul,
        dtypes=consts.FLOAT_DTYPES,
        # dtypes=attrs.FLOAT_DTYPES + attrs.COMPLEX_DTYPES,
    )
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
