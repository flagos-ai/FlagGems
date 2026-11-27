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


def _input_fn(shape, cur_dtype, device):
    inp = utils.generate_tensor_input(shape, cur_dtype, device)
    inp.view(-1)[0] = float("nan")
    if inp.numel() > 1:
        inp.view(-1)[1] = float("inf")
    if inp.numel() > 2:
        inp.view(-1)[2] = float("-inf")

    yield inp,


@pytest.mark.nan_to_num
def test_nan_to_num():
    bench = base.GenericBenchmark(
        op_name="nan_to_num",
        input_fn=_input_fn,
        torch_op=torch.nan_to_num,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


class NanToNumContiguousBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return super().set_more_shapes() + [
            (511,),
            (512,),
            (513,),
            (65536,),
            (65537,),
            (131072,),
            (1048577,),
        ]


@pytest.mark.nan_to_num
def test_nan_to_num_contiguous_launch():
    bench = NanToNumContiguousBenchmark(
        op_name="nan_to_num",
        input_fn=_input_fn,
        torch_op=torch.nan_to_num,
        dtypes=[torch.float32],
    )
    bench.run()
