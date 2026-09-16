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


def _scalar_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    yield inp, 0.5


@pytest.mark.gt
def test_gt():
    bench = base.BinaryPointwiseBenchmark(
        op_name="gt",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.gt_scalar
def test_gt_scalar():
    bench = base.GenericBenchmark(
        input_fn=_scalar_input_fn,
        op_name="gt_scalar",
        torch_op=torch.gt,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
