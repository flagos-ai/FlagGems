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


def _aminmax_raw_cpu(value):
    return torch.empty(0, dtype=value.dtype, device="cpu").set_(
        value.untyped_storage().cpu(),
        value.storage_offset(),
        value.shape,
        value.stride(),
    )


def _aminmax_input_snapshot(value):
    metadata = (
        value.device,
        value.dtype,
        tuple(value.shape),
        value.stride(),
        value.storage_offset(),
        value.untyped_storage()._cdata,
        value.requires_grad,
        value.untyped_storage().data_ptr(),
        value.untyped_storage().nbytes(),
        value.is_conj(),
        value.is_neg(),
    )
    contents = (
        _aminmax_raw_cpu(value).contiguous().reshape(-1).view(torch.uint8).clone()
    )
    return metadata, contents


def _checked_aminmax(inp):
    metadata, contents = _aminmax_input_snapshot(inp)
    try:
        result = flag_gems._aminmax(inp)
    finally:
        after_metadata, after_contents = _aminmax_input_snapshot(inp)
        assert after_metadata == metadata, "candidate changed input metadata"
        assert torch.equal(after_contents, contents), "candidate changed input values"
    assert isinstance(result, (tuple, list)) and len(result) == 2
    output_ranges = []
    for output in result:
        assert isinstance(output, torch.Tensor)
        assert output.device == metadata[0]
        assert output.dtype == metadata[1]
        assert output.shape == torch.Size([])
        assert output.untyped_storage()._cdata != metadata[5]
        storage = output.untyped_storage()
        if output.numel() and storage.nbytes():
            output_start = storage.data_ptr()
            output_end = output_start + storage.nbytes()
            if output.device == metadata[0] and metadata[8]:
                input_start = metadata[7]
                input_end = input_start + metadata[8]
                assert (
                    output_end <= input_start or input_end <= output_start
                ), "candidate output storage must not overlap input"
            output_ranges.append((output.device, output_start, output_end))
    assert result[0].untyped_storage()._cdata != result[1].untyped_storage()._cdata
    if len(output_ranges) == 2 and output_ranges[0][0] == output_ranges[1][0]:
        _, first_start, first_end = output_ranges[0]
        _, second_start, second_end = output_ranges[1]
        assert (
            first_end <= second_start or second_end <= first_start
        ), "candidate min/max output storage must not overlap"
    return result


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
    assert inp.dtype == dtype
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(res_min.detach().cpu(), ref_min.detach().cpu())
    utils.gems_assert_equal(res_max.detach().cpu(), ref_max.detach().cpu())


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_zero(shape, dtype):
    inp = torch.zeros(shape, dtype=dtype, device=flag_gems.device)
    assert inp.dtype == dtype
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(res_min.detach().cpu(), ref_min.detach().cpu())
    utils.gems_assert_equal(res_max.detach().cpu(), ref_max.detach().cpu())


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
def test__aminmax_inf(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp_flat = inp.float().flatten()
    inp_flat[0] = float("inf")
    inp_flat[1] = float("-inf")
    inp = inp_flat.reshape(shape).to(dtype)
    assert inp.dtype == dtype
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(
        res_min.detach().cpu(), ref_min.detach().cpu(), equal_nan=True
    )
    utils.gems_assert_equal(
        res_max.detach().cpu(), ref_max.detach().cpu(), equal_nan=True
    )


@pytest.mark.underscore_aminmax
@pytest.mark.parametrize("shape", UNDERSCORE_AMINMAX_SHAPES)
def test__aminmax_int(shape):
    inp = torch.randint(-100, 100, shape, dtype=torch.int32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(res_min.detach().cpu(), ref_min.detach().cpu())
    utils.gems_assert_equal(res_max.detach().cpu(), ref_max.detach().cpu())


@pytest.mark.underscore_aminmax
def test__aminmax_scalar():
    inp = torch.randn((), dtype=torch.float32, device=flag_gems.device)
    ref_inp = utils.to_reference(inp)

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(res_min.detach().cpu(), ref_min.detach().cpu())
    utils.gems_assert_equal(res_max.detach().cpu(), ref_max.detach().cpu())


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


def _aminmax_numeric_cpu(value):
    if value.dtype == torch.float64 and flag_gems.vendor_name == "enflame":
        result = _aminmax_raw_cpu(value)
        if value.is_conj():
            result = result.conj()
        if value.is_neg():
            result = result.neg()
        return result
    return value.detach().cpu()


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
    if dtype == torch.float64 and not utils.TO_CPU:
        if flag_gems.vendor_name == "enflame":
            pytest.skip("native Double uses Float storage on Enflame")
        if flag_gems.vendor_name == "kunlunxin":
            pytest.skip("native Double NaN reduction returns Float on Kunlunxin")
        if flag_gems.vendor_name == "iluvatar":
            pytest.skip("native Double NaN reduction returns zero on Iluvatar")
    n = 100000
    idx = {"first": 0, "middle": n // 2, "last": n - 1}[nan_pos]
    if dtype == torch.float64 and flag_gems.vendor_name in {"kunlunxin", "enflame"}:
        # The device Double factory silently constructs Float. Preserve the
        # original normal Double input and NaN position through raw bytes.
        host = torch.randn(n, dtype=dtype, device="cpu")
        host[idx] = float("nan")
        inp = host.view(torch.uint8).to(flag_gems.device).view(dtype)
    else:
        inp = torch.randn(n, dtype=dtype, device=flag_gems.device)
        inp[idx] = float("nan")
    assert inp.dtype == dtype
    ref_inp = (
        _aminmax_numeric_cpu(inp)
        if dtype == torch.float64 and flag_gems.vendor_name == "enflame"
        else utils.to_reference(inp)
    )

    ref_min, ref_max = torch._aminmax(ref_inp)
    res_min, res_max = _checked_aminmax(inp)

    assert torch.isnan(ref_min) and torch.isnan(ref_max), "reference must be NaN"
    utils.gems_assert_equal(
        _aminmax_numeric_cpu(res_min), ref_min.detach().cpu(), equal_nan=True
    )
    utils.gems_assert_equal(
        _aminmax_numeric_cpu(res_max), ref_max.detach().cpu(), equal_nan=True
    )


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
    res_min, res_max = _checked_aminmax(inp)

    utils.gems_assert_equal(res_min.detach().cpu(), ref_min.detach().cpu())
    utils.gems_assert_equal(res_max.detach().cpu(), ref_max.detach().cpu())


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
        _checked_aminmax(inp)
