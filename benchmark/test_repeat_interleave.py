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


class RepeatInterleaveBenchmark(base.GenericBenchmark):
    """
    Due to potential memory limitations, benchmark sizes need to be carefully controlled.

    Notably, when the input size is set to (1024, 1024, 1024) and the repeat dimensions
    are set to [1, 1, 2], the system encountered an "illegal memory access" error.
    To avoid such issues, we constrain the benchmark input sizes for these operations
    to prevent excessive memory usage.
    """

    DEFAULT_SHAPES = [
        (16, 256, 256),
        (128, 256, 256),
        (64, 64, 64, 64),
    ]

    def set_more_shapes(self):
        return []


# repeat_interleave.self_int(Tensor self, SymInt repeats,
# int? dim=None, *, SymInt? output_size=None) -> Tensor
def repeat_interleave_self_int_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    repeats = 3
    yield inp, repeats,


@pytest.mark.repeat_interleave_self_int
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
def test_repeat_interleave_self_int():
    bench = RepeatInterleaveBenchmark(
        input_fn=repeat_interleave_self_int_input_fn,
        op_name="repeat_interleave_self_int",
        torch_op=torch.repeat_interleave,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


# repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
def repeat_interleave_self_tensor_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    repeats = torch.randint(
        low=0,
        high=0x1F,  # control the repeats number here
        size=[
            shape[0],
        ],
        device=device,
    )
    dim = 0
    yield inp, repeats, dim


# @pytest.mark.skip(reason="This test case runs out of memory: issue #2674")
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro", reason="Issue #4131: not working"
)
@pytest.mark.repeat_interleave_self_tensor
def test_repeat_interleave_self_tensor():
    bench = RepeatInterleaveBenchmark(
        op_name="repeat_interleave_self_tensor",
        input_fn=repeat_interleave_self_tensor_input_fn,
        torch_op=torch.repeat_interleave,
        dtypes=[torch.int32],
    )
    bench.run()


# repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
def repeat_interleave_tensor_input_fn(shape, dtype, device):
    repeats = torch.randint(
        low=0,
        high=0x1F,  # control the repeats number here
        size=[
            shape[0],
        ],
        device=device,
    )
    yield repeats,


# @pytest.mark.skip(reason="This test case runs out of memory: issue #2674")
@pytest.mark.repeat_interleave_tensor
def test_repeat_interleave_tensor():
    bench = RepeatInterleaveBenchmark(
        op_name="repeat_interleave_tensor",
        input_fn=repeat_interleave_tensor_input_fn,
        torch_op=torch.repeat_interleave,
        dtypes=[torch.int32],
    )

    bench.run()
