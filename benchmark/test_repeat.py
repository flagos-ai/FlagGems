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

from . import base, consts, utils


class RepeatBenchmark(base.GenericBenchmark):
    """
    RepeatBenchmark designed to evaluate tensor repeat operations along specified dimensions.
    This includes operations like tile, repeat, and repeat_interval.
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    def set_more_shapes(self):
        return [(16, 256, 256), (512, 512, 512), (64, 64, 64, 64)]


def _input_fn(shape, dtype, device):
    inp1 = utils.generate_tensor_input(shape, dtype, device)
    inp2 = [1] * len(shape)
    inp2[0] = 2

    yield inp1, inp2,


@pytest.mark.repeat
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_repeat():
    bench = RepeatBenchmark(
        op_name="repeat",
        input_fn=_input_fn,
        torch_op=torch.Tensor.repeat,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
