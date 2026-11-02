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

from typing import Generator

import pytest
import torch

from . import base, consts

# Each entry is (input length, r, with_replacement), i.e. the output has
# C(n, r) (or C(n + r - 1, r)) rows and r columns. The sizes stay below a few
# million rows: r == 2 and r == 3 take the closed-form kernels, the higher
# ranks take the generic binary-search unranking kernel.
COMBINATIONS_CASES = [
    (1024, 2, False),
    (1024, 2, True),
    (256, 3, False),
    (256, 3, True),
    (64, 4, False),
    (32, 5, False),
    (16, 6, True),
]


class CombinationsBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = COMBINATIONS_CASES
        self.shape_desc = "input length, r, with_replacement"

    def get_input_iter(self, cur_dtype) -> Generator:
        for n, r, with_replacement in self.shapes:
            inp = base.generate_tensor_input((n,), cur_dtype, self.device)
            yield (inp, r, with_replacement)


@pytest.mark.combinations
def test_combinations():
    bench = CombinationsBenchmark(
        op_name="combinations",
        torch_op=torch.combinations,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
