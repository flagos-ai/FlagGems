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
    DIMS_LIST = [1]
    FLOAT_DTYPES = [torch.float32]
    KEEPDIM_DIMS_SHAPE = [(True, DIMS_LIST[0], utils.REDUCTION_SHAPES[0])]
else:
    DIMS_LIST = [0, 1, [0, 1], [1, 0]]
    FLOAT_DTYPES = utils.FLOAT_DTYPES
    KEEPDIM_DIMS_SHAPE = list(
        zip([True, False] * 2, DIMS_LIST, utils.REDUCTION_SHAPES + [(7, 4, 11, 1)])
    )


@pytest.mark.aminmax
@pytest.mark.parametrize("keepdim, dim, shape", KEEPDIM_DIMS_SHAPE)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_aminmax(shape, dim, keepdim, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    # torch.aminmax only supports single dim, use torch.amin/amax for multi-dim
    if isinstance(dim, list):
        ref_min = torch.amin(ref_inp, dim=dim, keepdim=keepdim)
        ref_max = torch.amax(ref_inp, dim=dim, keepdim=keepdim)
    else:
        ref_min, ref_max = torch.aminmax(ref_inp, dim=dim, keepdim=keepdim)
    # Call the FlagGems entry points directly rather than dispatching through
    # use_gems(): the framework handles dispatch, and check-kernelgen-tests
    # forbids use_gems() in tests for KernelGen operators.
    if isinstance(dim, list):
        res_min = flag_gems.amin(inp, dim=dim, keepdim=keepdim)
        res_max = flag_gems.amax(inp, dim=dim, keepdim=keepdim)
    else:
        res_min, res_max = flag_gems.aminmax(inp, dim=dim, keepdim=keepdim)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.aminmax
@pytest.mark.parametrize("shape", utils.REDUCTION_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test_accuracy_aminmax_no_dim(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch.aminmax(ref_inp)
    res_min, res_max = flag_gems.aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


# ---------------------------------------------------------------------------
# aten::_aminmax / aten::_aminmax.out
#
# The operator ids stay `_aminmax` / `_aminmax_out`, but pytest refuses to build a
# marker from an attribute starting with an underscore, so the markers are spelled
# `underscore_aminmax` / `underscore_aminmax_out`. The earlier approach of
# registering the names on pytest's private MarkGenerator has been dropped.
# ---------------------------------------------------------------------------

# Shapes for the whole-tensor _aminmax reduction.
UNDERSCORE_AMINMAX_SHAPES = utils.REDUCTION_SHAPES + [(1, 8192), (32, 50257)]


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_zero(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_inf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_flat = inp.float().flatten()
    inp_flat[0] = float("inf")
    inp_flat[1] = float("-inf")
    inp = inp_flat.reshape(shape).to(dtype)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min, equal_nan=True)
    utils.gems_assert_equal(res_max, ref_max, equal_nan=True)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
def test__aminmax_int(shape):
    inp = torch.randint(-100, 100, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax
def test__aminmax_scalar():
    inp = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax_out
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_out(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min = torch.empty((), dtype=dtype, device=ref_inp.device)
    ref_max = torch.empty((), dtype=dtype, device=ref_inp.device)
    torch.ops.aten._aminmax.out(ref_inp, out0=ref_min, out1=ref_max)

    min_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    max_out = torch.empty((), dtype=dtype, device=flag_gems.device)
    flag_gems._aminmax_out(inp, out0=min_out, out1=max_out)

    utils.gems_assert_equal(min_out, ref_min)
    utils.gems_assert_equal(max_out, ref_max)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16])
@pytest.mark.parametrize("nan_pos", ["first", "middle", "last"])
def test_underscore_aminmax_nan(dtype, nan_pos):
    """A single NaN forces *both* outputs to NaN, as torch._aminmax does.

    tl.min / tl.max drop NaNs rather than propagating them, so this needs an
    explicit check in each reduction stage. The tensor is large enough that the
    NaN has to survive both stages, and the position is varied so a NaN in a
    non-first tile is covered too.
    """
    n = 100000
    inp = torch.randn(n, dtype=dtype, device=flag_gems.device)
    idx = {"first": 0, "middle": n // 2, "last": n - 1}[nan_pos]
    inp[idx] = float("nan")
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    assert torch.isnan(ref_min) and torch.isnan(ref_max), "reference must be NaN"
    utils.gems_assert_equal(res_min, ref_min, equal_nan=True)
    utils.gems_assert_equal(res_max, ref_max, equal_nan=True)


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize(
    "view", ["slice", "transpose", "narrow", "expand", "broadcast"]
)
def test_underscore_aminmax_non_contiguous(view):
    """Non-dense views must be read in the right order.

    The kernels index with a flat offset, so a sliced or transposed view would
    otherwise read the wrong elements, and an expanded view (stride 0) would read
    past its own storage.
    """
    base = torch.randn(8, 12, dtype=torch.float32, device=flag_gems.device)
    inp = {
        "slice": base[:, ::2],
        "transpose": base.t(),
        "narrow": base[2:6, 3:9],
        "expand": base[:1, :1].expand(6, 9),
        "broadcast": base[:1].expand(8, 12),
    }[view]
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = flag_gems._aminmax(inp)

    utils.gems_assert_equal(res_min, ref_min)
    utils.gems_assert_equal(res_max, ref_max)


@pytest.mark.underscore_aminmax_out
def test_underscore_aminmax_out_resizes_and_aliases():
    """Mis-shaped outputs are resized to scalars and the caller's objects returned."""
    inp = torch.randn(64, dtype=torch.float32, device=flag_gems.device)
    ref_min, ref_max = torch._aminmax(utils.to_reference(inp))

    out0 = torch.empty(5, dtype=torch.float32, device=flag_gems.device)
    out1 = torch.empty(3, 7, dtype=torch.float32, device=flag_gems.device)
    res0, res1 = flag_gems._aminmax_out(inp, out0=out0, out1=out1)

    assert res0 is out0 and res1 is out1, "returned tensors must alias out0/out1"
    assert out0.shape == torch.Size([]) and out1.shape == torch.Size([])
    utils.gems_assert_equal(out0, ref_min)
    utils.gems_assert_equal(out1, ref_max)


@pytest.mark.underscore_aminmax_out
@pytest.mark.parametrize("bad", ["dtype", "device"])
def test_underscore_aminmax_out_rejects_mismatch(bad):
    """A dtype or device mismatch on the outputs raises, as in ATen."""
    inp = torch.randn(32, dtype=torch.float32, device=flag_gems.device)
    if bad == "dtype":
        out0 = torch.empty((), dtype=torch.float64, device=flag_gems.device)
        out1 = torch.empty((), dtype=torch.float32, device=flag_gems.device)
        match = "dtype"
    else:
        if utils.TO_CPU:
            pytest.skip("device mismatch needs a non-CPU device under test")
        out0 = torch.empty((), dtype=torch.float32, device="cpu")
        out1 = torch.empty((), dtype=torch.float32, device="cpu")
        match = "device"
    with pytest.raises(RuntimeError, match=match):
        flag_gems._aminmax_out(inp, out0=out0, out1=out1)


@pytest.mark.underscore_aminmax
def test_underscore_aminmax_empty_matches_aten():
    """An empty input is rejected, matching ATen's "no identity" error."""
    inp = torch.empty(0, dtype=torch.float32, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems._aminmax(inp)
