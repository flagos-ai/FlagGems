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

# torch.cosine_similarity computes <x1, x2> / (max(||x1||_2, eps) * max(||x2||_2, eps))
# reduced over `dim`, broadcasting x1 and x2 to a common shape first. The gems
# kernel is expected to match torch across dtypes, reduction dims, broadcasting,
# and ndim >= 3 (every leading/trailing dim outside `dim` is a batch row).
if cfg.QUICK_MODE:
    # --quick smoke run restricts to fp32; full runs use utils.FLOAT_DTYPES.
    FLOAT_DTYPES = [torch.float32]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES

# (shape, dim) pairs: reduce over the given dim; output drops that dim.
SHAPE_DIM = [
    ((7,), 0),  # 1-D: single pair of D-dim vectors -> scalar output
    ((64, 64), 1),
    ((64, 64), 0),
    ((1024, 257), 1),
    ((16, 32, 64), -1),
    ((16, 32, 64), 1),
    ((8, 4, 8, 16), 2),
]


@pytest.mark.cosine_similarity
@pytest.mark.parametrize("shape, dim", SHAPE_DIM)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cosine_similarity_accuracy(shape, dim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = torch.cosine_similarity(ref_x1, ref_x2, dim=dim, eps=1e-8)
    res_out = flag_gems.cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    reduce_dim = shape[dim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


# (x1_shape, x2_shape, dim): torch broadcasts x1/x2 to a common shape, then
# reduces over `dim`. Requires the op to broadcast internally.
BROADCAST_SHAPES = [
    ((3, 4), (1, 4), 1),  # row-broadcast
    ((3, 1, 5), (1, 4, 5), 2),  # 3-D two-sided broadcast -> out (3, 4)
    ((2, 3, 4, 5), (5,), 3),  # 4-D vs trailing 1-D -> out (2, 3, 4)
    ((3, 4), (4,), 1),  # 2-D vs trailing 1-D
]


@pytest.mark.cosine_similarity
@pytest.mark.parametrize("x1_shape, x2_shape, dim", BROADCAST_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cosine_similarity_broadcast(x1_shape, x2_shape, dim, dtype):
    torch.manual_seed(0)
    x1 = torch.randn(x1_shape, dtype=dtype, device=flag_gems.device)
    x2 = torch.randn(x2_shape, dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = torch.cosine_similarity(ref_x1, ref_x2, dim=dim, eps=1e-8)
    res_out = flag_gems.cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    reduce_dim = torch.broadcast_shapes(x1_shape, x2_shape)[dim]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


# Near-zero vectors exercise the eps clamp: each L2 norm is clamped to eps before
# division, so a zero row yields 0 (not NaN).
@pytest.mark.cosine_similarity
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_cosine_similarity_eps_clamp(dtype):
    torch.manual_seed(0)
    x1 = torch.zeros((4, 8), dtype=dtype, device=flag_gems.device)
    x2 = torch.randn((4, 8), dtype=dtype, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = torch.cosine_similarity(ref_x1, ref_x2, dim=1, eps=1e-8)
    res_out = flag_gems.cosine_similarity(x1, x2, dim=1, eps=1e-8)

    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=8)


# float64 accuracy: verify fp64 inputs are accumulated in fp64 (not silently
# downcast to fp32, which would lose ~9 digits of precision).
FP64_SHAPE_DIM = [((64, 257), 1), ((16, 32, 64), -1)]


@pytest.mark.cosine_similarity
@pytest.mark.parametrize("shape, dim", FP64_SHAPE_DIM)
def test_cosine_similarity_fp64(shape, dim):
    if not utils.fp64_is_supported:
        pytest.skip("fp64 not supported on this device")
    torch.manual_seed(0)
    x1 = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    x2 = torch.randn(shape, dtype=torch.float64, device=flag_gems.device)
    ref_x1 = utils.to_reference(x1, True)
    ref_x2 = utils.to_reference(x2, True)

    ref_out = torch.cosine_similarity(ref_x1, ref_x2, dim=dim, eps=1e-8)
    res_out = flag_gems.cosine_similarity(x1, x2, dim=dim, eps=1e-8)

    utils.gems_assert_close(res_out, ref_out, torch.float64, reduce_dim=shape[dim])
