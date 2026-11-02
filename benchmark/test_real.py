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

from . import base, consts

REAL_SHAPES = [(1,), (1024, 1024), (4096, 4096)]


class RealBenchmark(base.Benchmark):
    is_conj = False

    def set_shapes(self, shape_file_path=None):
        self.shapes = REAL_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            input = torch.empty(shape, dtype=cur_dtype, device=self.device)
            yield (input.conj() if self.is_conj else input,)


@pytest.mark.real
@pytest.mark.parametrize(
    ("op_name", "is_conj"),
    (("real", False), ("real_conj", True)),
    ids=("plain", "conj"),
)
def test_real(op_name, is_conj):
    # This measures dispatch and view-metadata overhead; no device kernel runs.
    bench = RealBenchmark(
        op_name=op_name,
        torch_op=torch.real,
        dtypes=consts.COMPLEX_DTYPES,
        is_conj=is_conj,
    )
    bench.run()
