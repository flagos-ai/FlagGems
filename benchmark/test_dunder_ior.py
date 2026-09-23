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

from . import base, consts, utils


@pytest.mark.dunder_ior_tensor
def test_dunder_ior_inplace():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_ior_tensor",
        torch_op=lambda a, b: a.__ior__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()


def _input_fn_scalar(shape, cur_dtype, device):
    inp1 = utils.generate_tensor_input(shape, cur_dtype, device)
    if cur_dtype == consts.BOOL_DTYPES[0]:
        inp2 = True
    else:
        inp2 = 0x00FF
    yield inp1, inp2


@pytest.mark.dunder_ior_scalar
def test_dunder_ior_scalar_inplace():
    bench = base.GenericBenchmark(
        op_name="dunder_ior_scalar",
        input_fn=_input_fn_scalar,
        torch_op=lambda a, b: a.__ior__(b),
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
        is_inplace=True,
    )
    bench.run()
