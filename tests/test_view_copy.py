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

from . import accuracy_utils as utils
from . import conftest as cfg

# Shapes covering various reshape patterns: flatten, identity, merge dims, high-rank
if cfg.QUICK_MODE:
    VIEW_COPY_SHAPES = [
        ((2, 19, 7), (266,)),
        ((20, 320, 15), (20, 4800)),
    ]
else:
    VIEW_COPY_SHAPES = [
        ((2, 19, 7), (266,)),
        ((1,), ()),
        ((1024, 1024), (1048576,)),
        ((1024, 1024), (1024, 1024)),
        ((20, 320, 15), (20, 4800)),
        ((20, 320, 15), (6400, 15)),
        ((16, 128, 64, 60), (16, 128, 3840)),
        ((16, 7, 57, 32, 29), (16, 7, 57, 928)),
    ]


@pytest.mark.view_copy
@pytest.mark.parametrize("source_shape, target_shape", VIEW_COPY_SHAPES)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_view_copy(source_shape, target_shape, dtype):
    # Create input with the source shape
    inp = torch.randn(source_shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.view_copy(ref_inp, target_shape)
    with flag_gems.use_gems():
        res_out = torch.view_copy(inp, target_shape)

    utils.gems_assert_equal(res_out, ref_out)
