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

from . import base, consts, utils


class EmbeddingBenchmark(base.GenericBenchmark2DOnly):
    def set_more_shapes(self):
        return []


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield {"input": inp},

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        yield {"input": inp, "offset": 1, "dim1": 0, "dim2": -1},


@pytest.mark.diag_embed
def test_diag_embed():
    bench = EmbeddingBenchmark(
        op_name="diag_embed",
        input_fn=_input_fn,
        torch_op=torch.diag_embed,
        dtypes=consts.FLOAT_DTYPES + consts.INT_DTYPES + consts.BOOL_DTYPES,
    )

    bench.run()
