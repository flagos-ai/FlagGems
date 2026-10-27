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


class TileBenchmark(base.GenericBenchmark):
    """
    TileBenchmark designed to evaluate tensor repeat operations along specified dimensions.
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    def set_more_shapes(self):
        more_shapes = [
            (16, 256, 256),
            (512, 512, 512),
            (64, 64, 64, 64),
        ]
        return more_shapes


def _input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    dim = [1] * len(shape)
    dim[0] = 2
    yield inp, {"dims": dim}


@pytest.mark.tile
def test_tile():
    bench = TileBenchmark(
        op_name="tile",
        input_fn=_input_fn,
        torch_op=torch.tile,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
