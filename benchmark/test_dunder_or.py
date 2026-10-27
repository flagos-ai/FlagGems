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


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0x00FF


@pytest.mark.dunder_or_tensor
def test_dunder_or_tensor():
    bench = base.BinaryPointwiseBenchmark(
        op_name="dunder_or_tensor",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()


@pytest.mark.dunder_or_scalar
def test_dunder_or_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="dunder_or_scalar",
        torch_op=lambda a, b: a | b,
        dtypes=consts.INT_DTYPES + consts.BOOL_DTYPES,
    )
    bench.run()
