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

from typing import Generator

import pytest
import torch

from . import base, consts, utils


class BitwiseRightShiftBenchmark(base.Benchmark):
    def set_more_shapes(self):
        special_shapes_2d = [(1024, 2**i) for i in range(0, 20, 4)]
        sp_shapes_3d = [(64, 64, 2**i) for i in range(0, 15, 4)]
        return special_shapes_2d + sp_shapes_3d

    def get_input_iter(self, dtype) -> Generator:
        for shape in self.shapes:
            inp1 = utils.generate_tensor_input(shape, dtype, self.device)
            shift_amount = torch.randint(0, 8, shape, dtype=dtype, device="cpu").to(
                self.device
            )
            yield inp1, shift_amount


@pytest.mark.bitwise_right_shift
def test_bitwise_right_shift():
    bench = BitwiseRightShiftBenchmark(
        op_name="bitwise_right_shift",
        torch_op=torch.bitwise_right_shift,
        dtypes=consts.INT_DTYPES,
    )

    bench.run()
