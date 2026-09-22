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


pytestmark = pytest.mark.skipif(
    flag_gems.vendor_name != "hygon",
    reason="Hygon MV correctness test requires the Hygon backend",
)


_HYGON_COMMON_ROWS = [
    1,
    2,
    4,
    8,
    16,
    *range(24, 257, 8),
    *range(272, 513, 16),
]

HYGON_MV_REAL_MODEL_SHAPES = [
    *[(n, 2048) for n in _HYGON_COMMON_ROWS],
    (1034, 2048),
    (1035, 2048),
    (1036, 2048),
    (1352, 2048),
    (2048, 2048),
    (3104, 2048),
    (4107, 2048),
    (4108, 2048),
    (4138, 2048),
    (4434, 2048),
    (4435, 2048),
    (4436, 2048),
    (9309, 2048),
    (9330, 2048),
    (11469, 2048),
    (13421, 2048),
    (13422, 2048),
    (16384, 2048),
    *[(n, 4096) for n in _HYGON_COMMON_ROWS],
    (1035, 4096),
    (1036, 4096),
    (2048, 4096),
    (4107, 4096),
    (4108, 4096),
    (4434, 4096),
    (4435, 4096),
    (5240, 4096),
    (8214, 4096),
    (13421, 4096),
    (13422, 4096),
    (16384, 4096),
]

HYGON_MV_QUICK_SHAPES = [
    (1, 2048),
    (16, 2048),
    (128, 2048),
    (512, 2048),
    (1034, 2048),
    (16384, 2048),
    (1, 4096),
    (16, 4096),
    (128, 4096),
    (512, 4096),
    (1035, 4096),
    (16384, 4096),
]

assert len(HYGON_MV_REAL_MODEL_SHAPES) == 132
assert len(HYGON_MV_QUICK_SHAPES) == 12

if cfg.QUICK_MODE:
    _SHAPES = HYGON_MV_QUICK_SHAPES
    _DTYPES = [torch.float32]
else:
    _SHAPES = HYGON_MV_REAL_MODEL_SHAPES
    _DTYPES = utils.FLOAT_DTYPES


@pytest.mark.mv
@pytest.mark.parametrize("M, N", [(k, n) for n, k in _SHAPES])
@pytest.mark.parametrize("dtype", _DTYPES)
def test_hygon_mv(M, N, dtype):
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)
    ref_matrix = utils.to_reference(matrix, True)
    ref_vector = utils.to_reference(vector, True)

    ref_out = torch.mv(ref_matrix, ref_vector)
    with flag_gems.use_gems():
        res_out = torch.mv(matrix, vector)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=M)
