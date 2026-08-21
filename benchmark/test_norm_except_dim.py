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


def norm_except_dim_input_fn(shape, dtype, device):
    inp = utils.generate_tensor_input(shape, dtype, device)
    # pow=2, dim=0 are the torch defaults; yield the tensor alone so the norm
    # keeps the leading dim and reduces the rest.
    yield inp,


@pytest.mark.norm_except_dim
def test_norm_except_dim():
    bench = base.GenericBenchmark2DOnly(
        input_fn=norm_except_dim_input_fn,
        op_name="norm_except_dim",
        torch_op=torch.norm_except_dim,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
