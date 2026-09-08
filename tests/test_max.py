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

if cfg.QUICK_MODE:
    FLOAT_DTYPES = [torch.float32]
    DIM_LIST = [0]
    KEEPDIM = [True]
else:
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    DIM_LIST = [0, 1]
    KEEPDIM = [True, False]

NAN_FLOAT_DTYPES = [torch.float32] if cfg.QUICK_MODE else utils.ALL_FLOAT_DTYPES


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_max_all_neg_inf(shape, dtype):
    inp = torch.full(
        shape, fill_value=float("-inf"), dtype=dtype, device=flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.max
@pytest.mark.parametrize("dtype", NAN_FLOAT_DTYPES)
@pytest.mark.parametrize("nan_case", ["first", "middle", "last", "multiple", "all"])
def test_max_with_nan(dtype, nan_case):
    # A non-power-of-two size exercises both stages of the global reduction.
    width = 4097
    inp = torch.arange(width, dtype=dtype, device=flag_gems.device)
    nan_indices = {
        "first": [0],
        "middle": [width // 2],
        "last": [width - 1],
        "multiple": [17, width // 2, width - 1],
    }
    if nan_case == "all":
        inp.fill_(float("nan"))
    else:
        inp[nan_indices[nan_case]] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.max
@pytest.mark.parametrize(
    "width",
    [
        pytest.param(65537, id="cambricon-multi-cta"),
        pytest.param(48 * 16384 + 1, id="gcu400-grid-stride"),
        pytest.param(10 * 1024 * 1024 + 1, id="gcu400-simple"),
    ],
)
def test_max_with_nan_large_input(width):
    # Cover vendor-specific multi-CTA, grid-stride, and large-input kernels.
    inp = torch.zeros(width, dtype=torch.float32, device=flag_gems.device)
    inp[-1] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out, equal_nan=True)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES + [[1]])
@pytest.mark.parametrize("dtype", utils.ALL_INT_DTYPES)
def test_max_int(shape, dtype):
    inp = torch.randint(-1000, 1000, shape, dtype=dtype, device="cpu").to(
        flag_gems.device
    )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


@pytest.mark.max
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_uncontiguous(shape, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device="cpu")[::2, ::2].to(
            flag_gems.device
        )
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu")[
            ::2, ::2
        ].to(flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp)
    with flag_gems.use_gems():
        res_out = torch.max(inp)

    utils.gems_assert_equal(res_out, ref_out)


# Issue #2831: failed at (200, 40999, 3), while successed at this shape in mean_dim
@pytest.mark.max_dim
@pytest.mark.parametrize("shape", utils.REDUCTION_SMALL_SHAPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize("dim", DIM_LIST)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)


@pytest.mark.max_dim
@pytest.mark.parametrize("dtype", NAN_FLOAT_DTYPES)
@pytest.mark.parametrize("keepdim", KEEPDIM)
@pytest.mark.parametrize(
    "dim",
    [
        pytest.param(0, id="first"),
        pytest.param(1, id="middle"),
        pytest.param(2, id="last"),
    ],
)
def test_max_dim_with_nan(dtype, keepdim, dim):
    # Moving the same six reduction slices to every dimension exercises the
    # backend-specific first/middle/last-dimension kernels. A width just above
    # 4096 also crosses their common reduction tile boundaries.
    width = 4097
    base = torch.zeros((6, width), dtype=dtype, device=flag_gems.device)
    base[:, -1] = 1
    base[0, 0] = float("nan")
    base[1, width // 2] = float("nan")
    base[2, width - 1] = float("nan")
    base[3, [17, width // 2, width - 1]] = float("nan")
    base[4].fill_(float("nan"))
    inp = base.reshape(2, 3, width).movedim(-1, dim).contiguous()
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp, dim=dim, keepdim=keepdim)
    with flag_gems.use_gems():
        res_out = torch.max(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out.indices, ref_out.indices)
    utils.gems_assert_equal(res_out.values, ref_out.values, equal_nan=True)


@pytest.mark.max_dim
@pytest.mark.parametrize("dtype", NAN_FLOAT_DTYPES)
@pytest.mark.parametrize("width", [1, 7])
def test_max_dim_with_nan_small(dtype, width):
    # width=1 covers singleton fast paths; width=7 covers MThreads' small kernel.
    if width == 1:
        inp = torch.tensor(
            [[float("nan")], [1.0]], dtype=dtype, device=flag_gems.device
        )
    else:
        inp = torch.tensor(
            [
                [float("nan"), 2.0, float("nan"), float("inf"), -1.0, 3.0, 3.0],
                [float("-inf")] * width,
                [1.0, float("inf"), float("inf"), 0.0, -1.0, 3.0, 3.0],
                [3.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0],
            ],
            dtype=dtype,
            device=flag_gems.device,
        )
    ref_inp = utils.to_reference(inp)

    ref_out = torch.max(ref_inp, dim=1)
    with flag_gems.use_gems():
        res_out = torch.max(inp, dim=1)

    utils.gems_assert_equal(res_out.indices, ref_out.indices)
    utils.gems_assert_equal(res_out.values, ref_out.values, equal_nan=True)


@pytest.mark.max_dim
@pytest.mark.skipif(
    flag_gems.vendor_name == "aipu", reason="Issue #3009: Big shape run slowly."
)
@pytest.mark.parametrize("shape", [(4, 1048577, 4)])
@pytest.mark.parametrize("keepdim, dim", [(True, 1), (False, 1)])
@pytest.mark.parametrize("dtype", FLOAT_DTYPES + utils.ALL_INT_DTYPES)
def test_max_dim_big_shape(shape, dim, keepdim, dtype):
    if dtype in FLOAT_DTYPES:
        inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    else:
        inp = torch.randint(-10000, 10000, shape, dtype=dtype, device="cpu").to(
            flag_gems.device
        )
    ref_inp = utils.to_reference(inp)

    ref_out_value, ref_out_index = torch.max(ref_inp, dim=dim, keepdim=keepdim)

    with flag_gems.use_gems():
        res_out_value, res_out_index = torch.max(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_out_index, ref_out_index)
    utils.gems_assert_equal(res_out_value, ref_out_value)
