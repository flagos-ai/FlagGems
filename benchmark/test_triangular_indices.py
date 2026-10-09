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

pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name == "ascend",
    reason=(
        "Ascend native torch.tril_indices/triu_indices benchmark baseline "
        "is prohibitively slow."
    ),
)


class TriangularIndicesBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (64, 64),
            (256, 1024),
            (1024, 256),
            (1024, 1024),
        ]
        self.shape_desc = "row, col"

    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    row, col = shape
    yield {
        "row": row,
        "col": col,
        "offset": 0,
        "dtype": dtype,
        "device": device,
    },

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        diagonal = min(row, col) // 4
        for offset in (-diagonal, diagonal):
            yield {
                "row": row,
                "col": col,
                "offset": offset,
                "dtype": dtype,
                "device": device,
            },


@pytest.mark.tril_indices
def test_tril_indices():
    bench = TriangularIndicesBenchmark(
        op_name="tril_indices",
        input_fn=_input_fn,
        torch_op=torch.tril_indices,
        gems_op=flag_gems.tril_indices,
        dtypes=[torch.int32, torch.int64],
    )
    bench.run()


@pytest.mark.triu_indices
def test_triu_indices():
    bench = TriangularIndicesBenchmark(
        op_name="triu_indices",
        input_fn=_input_fn,
        torch_op=torch.triu_indices,
        gems_op=flag_gems.triu_indices,
        dtypes=[torch.int32, torch.int64],
    )
    bench.run()
