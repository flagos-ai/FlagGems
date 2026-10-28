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


@pytest.fixture(autouse=True)
def _seed_random_inputs():
    torch.manual_seed(0)


DTYPES = [torch.float32] if cfg.QUICK_MODE else [torch.float32, torch.float64]
INTERPOLATIONS = (
    ["linear"]
    if cfg.QUICK_MODE
    else [
        "linear",
        "lower",
        "higher",
        "nearest",
        "midpoint",
    ]
)


def _input(shape, dtype):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    flat = inp.reshape(-1)
    flat[::5] = float("nan")
    return inp


def _native_reference(
    inp, q, dim=None, keepdim=False, *, interpolation="linear", out=None
):
    # The configured Enflame native path requests unsupported Long sort indices.
    # Keep the original promoted arithmetic and reference output ABI on CPU.
    tensor_q = isinstance(q, torch.Tensor)
    overload = (
        ("out" if out is not None else "default")
        if tensor_q
        else ("scalar_out" if out is not None else "scalar")
    )
    op = getattr(torch.ops.aten.nanquantile, overload)
    if flag_gems.vendor_name != "enflame":
        kwargs = {"interpolation": interpolation}
        if out is not None:
            kwargs["out"] = out
        return op(inp, q, dim, keepdim, **kwargs)
    cpu_inp = inp.to("cpu")
    cpu_q = q.to("cpu") if tensor_q else q
    kwargs = {"interpolation": interpolation}
    if out is not None:
        kwargs["out"] = torch.empty_like(out, device="cpu")
    result = op(cpu_inp, cpu_q, dim, keepdim, **kwargs)
    assert result.dtype == inp.dtype and result.device.type == "cpu"
    result = result.to(inp.device)
    assert result.device == inp.device and result.dtype == inp.dtype
    if out is not None:
        assert out.device == inp.device and out.dtype == inp.dtype
        out.resize_(result.shape)
        out.copy_(result)
        return out
    return result


def _input_abi(inp, q):
    tensors = [x for x in (inp, q) if isinstance(x, torch.Tensor)]
    return [
        (x, x.device, x.dtype, tuple(x.shape), x.stride(), x.data_ptr())
        for x in tensors
    ]


def _assert_output_abi(result, saved, expected_shape, *, out=None):
    assert isinstance(result, torch.Tensor)
    assert result.device == saved[0][1]
    assert result.dtype == saved[0][2]
    assert tuple(result.shape) == tuple(expected_shape)
    if out is not None:
        assert result is out
    for inp, device, dtype, shape, stride, pointer in saved:
        assert inp.device == device and inp.dtype == dtype
        assert tuple(inp.shape) == shape and inp.stride() == stride
        assert inp.data_ptr() == pointer
        if out is None and inp.numel() and result.numel():
            assert (
                result.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
            )


def _assert_close(result, reference, dtype, reduction_size):
    utils.gems_assert_close(
        result,
        reference,
        dtype,
        reduce_dim=reduction_size,
        equal_nan=True,
    )


@pytest.mark.nanquantile
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("dim, keepdim", [(None, False), (1, False), (-1, True)])
def test_nanquantile_tensor(dtype, interpolation, dim, keepdim):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    inp = _input((7, 33), dtype).T
    q = torch.tensor([0.0, 0.2, 0.5, 0.8, 1.0], dtype=dtype, device=inp.device)
    ref_inp, ref_q = utils.to_reference(inp, True), utils.to_reference(q, True)
    reference = _native_reference(
        ref_inp, ref_q, dim, keepdim, interpolation=interpolation
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(
        inp, q, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    reduction_size = inp.numel() if dim is None else inp.shape[dim]
    _assert_output_abi(result, saved_abi, reference.shape)
    _assert_close(result, reference, dtype, reduction_size)


@pytest.mark.nanquantile_scalar
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
@pytest.mark.parametrize("dim, keepdim", [(None, False), (0, True), (-1, False)])
def test_nanquantile_scalar(dtype, interpolation, dim, keepdim):
    inp = _input((17, 19), dtype)
    ref_inp = utils.to_reference(inp, True)
    reference = torch.ops.aten.nanquantile.scalar(
        ref_inp, 0.37, dim, keepdim, interpolation=interpolation
    )
    result = flag_gems.nanquantile_scalar(
        inp, 0.37, dim=dim, keepdim=keepdim, interpolation=interpolation
    )
    reduction_size = inp.numel() if dim is None else inp.shape[dim]
    _assert_close(result, reference, dtype, reduction_size)


@pytest.mark.nanquantile_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("q_shape", [(), (3,), (0,)])
def test_nanquantile_out(dtype, q_shape):
    inp = _input((5, 13), dtype)
    q = torch.empty(q_shape, dtype=dtype, device=inp.device)
    if q.numel():
        values = torch.linspace(0.1, 0.9, q.numel(), dtype=dtype, device=inp.device)
        q.copy_(values.reshape(q_shape))
    ref_inp, ref_q = utils.to_reference(inp, True), utils.to_reference(q, True)
    ref_out = torch.empty(0, dtype=ref_inp.dtype, device=ref_inp.device)
    reference = torch.ops.aten.nanquantile.out(
        ref_inp, ref_q, 1, True, interpolation="linear", out=ref_out
    )
    out = torch.empty(0, dtype=dtype, device=inp.device)
    result = flag_gems.nanquantile_out(
        inp, q, dim=1, keepdim=True, interpolation="linear", out=out
    )
    assert result is out
    _assert_close(result, reference, dtype, inp.shape[1])


def _scalar_out_target_from_cpu(inp):
    # Kunlunxin's ordinary Float64 factory/transfer silently narrows storage.
    if flag_gems.vendor_name == "kunlunxin" and inp.dtype == torch.float64:
        return (
            inp.contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .to(flag_gems.device)
            .view(inp.dtype)
            .reshape(inp.shape)
        )
    return inp.to(flag_gems.device)


def _scalar_out_input(shape, dtype):
    if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64:
        inp = torch.randn(shape, dtype=dtype, device="cpu")
        inp.reshape(-1)[::5] = float("nan")
        return _scalar_out_target_from_cpu(inp)
    return _input(shape, dtype)


@pytest.mark.nanquantile_scalar_out
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("interpolation", INTERPOLATIONS)
def test_nanquantile_scalar_out(dtype, interpolation):
    if flag_gems.vendor_name == "enflame" and dtype == torch.float64:
        pytest.skip("Enflame Float64 storage loses adjacent-double values")
    if dtype == torch.float64 and not cfg.TO_CPU:
        if flag_gems.vendor_name == "kunlunxin":
            pytest.skip(
                "Kunlunxin native scalar nanquantile creates a mismatched Float64 q tensor"
            )
        if flag_gems.vendor_name == "ascend":
            pytest.skip("Ascend native nanquantile sorting does not support Float64")
    inp = _scalar_out_input((11, 15), dtype)
    assert inp.dtype == dtype and inp.device.type == torch.device(flag_gems.device).type
    ref_inp = utils.to_reference(inp, True)
    ref_out = torch.empty(0, dtype=ref_inp.dtype, device=ref_inp.device)
    reference = _native_reference(
        ref_inp, 0.73, 0, False, interpolation=interpolation, out=ref_out
    )
    out = (
        _scalar_out_target_from_cpu(torch.empty((1,), dtype=dtype, device="cpu"))
        if flag_gems.vendor_name == "kunlunxin" and dtype == torch.float64
        else torch.empty((1,), dtype=dtype, device=inp.device)
    )
    assert out.dtype == dtype and out.device == inp.device
    saved_abi = _input_abi(inp, 0.73)
    result = flag_gems.nanquantile_scalar_out(
        inp, 0.73, dim=0, interpolation=interpolation, out=out
    )
    _assert_output_abi(result, saved_abi, reference.shape, out=out)
    assert result is out
    _assert_close(result, reference, dtype, inp.shape[0])


@pytest.mark.parametrize(
    "tensor_q,use_out",
    [
        pytest.param(False, False, marks=pytest.mark.nanquantile_scalar),
        pytest.param(True, False, marks=pytest.mark.nanquantile),
        pytest.param(False, True, marks=pytest.mark.nanquantile_scalar_out),
        pytest.param(True, True, marks=pytest.mark.nanquantile_out),
    ],
)
def test_nanquantile_dim_none_keepdim_shape(tensor_q, use_out):
    inp = _input((2, 3), torch.float32)
    ref_inp = utils.to_reference(inp, True)
    q = torch.tensor([0.25, 0.75], device=inp.device) if tensor_q else 0.25
    ref_q = utils.to_reference(q, True) if tensor_q else q
    if use_out:
        out = torch.empty(0, device=inp.device)
        ref_out = torch.empty(0, dtype=ref_inp.dtype, device=ref_inp.device)
        if tensor_q:
            reference = torch.ops.aten.nanquantile.out(
                ref_inp, ref_q, None, True, interpolation="linear", out=ref_out
            )
            result = flag_gems.nanquantile_out(inp, q, dim=None, keepdim=True, out=out)
        else:
            reference = _native_reference(
                ref_inp, q, None, True, interpolation="linear", out=ref_out
            )
            saved_abi = _input_abi(inp, q)
            result = flag_gems.nanquantile_scalar_out(
                inp, q, dim=None, keepdim=True, out=out
            )
        assert result is out
        assert reference is ref_out
    elif tensor_q:
        reference = _native_reference(ref_inp, ref_q, keepdim=True)
        saved_abi = _input_abi(inp, q)
        result = flag_gems.nanquantile(inp, q, keepdim=True)
    else:
        reference = torch.nanquantile(ref_inp, q, keepdim=True)
        result = flag_gems.nanquantile_scalar(inp, q, keepdim=True)
    if (tensor_q and not use_out) or (not tensor_q and use_out):
        _assert_output_abi(
            result, saved_abi, reference.shape, out=out if use_out else None
        )
    assert result.shape == reference.shape
    _assert_close(result, reference, torch.float32, inp.numel())


@pytest.mark.nanquantile
@pytest.mark.parametrize("valid_count", [0, 1])
def test_nanquantile_maximum_reduction_one_valid(valid_count):
    # Exercise the binary-search upper bound through the complete public path.
    size = 1 << 24
    inp = torch.full((size,), float("nan"), device=flag_gems.device)
    if valid_count:
        inp[0] = 13.0
    q = torch.tensor([0.0, 0.5, 1.0], device=inp.device)
    reference = _native_reference(
        utils.to_reference(inp, True), utils.to_reference(q, True)
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(inp, q)
    _assert_output_abi(result, saved_abi, reference.shape)
    assert result.dtype == inp.dtype
    assert result.device == inp.device
    assert result.shape == reference.shape
    utils.gems_assert_equal(result.to(reference.dtype), reference, equal_nan=True)


@pytest.mark.nanquantile
@pytest.mark.parametrize("size", [1025, 2049])
def test_nanquantile_large_q_is_tiled(size):
    inp = _input((2, size), torch.float32)
    q = torch.linspace(0, 1, 513, device=inp.device)
    reference = _native_reference(
        utils.to_reference(inp, True), utils.to_reference(q, True), dim=1
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(inp, q, dim=1)
    _assert_output_abi(result, saved_abi, reference.shape)
    _assert_close(result, reference, torch.float32, size)


@pytest.mark.nanquantile
@pytest.mark.parametrize(
    "interpolation", ["linear", "lower", "higher", "nearest", "midpoint"]
)
def test_nanquantile_gather_interpolations(interpolation):
    inp = _input((2, 4097), torch.float32)
    q = torch.tensor([0.13, 0.51, 0.92], device=inp.device)
    reference = _native_reference(
        utils.to_reference(inp, True),
        utils.to_reference(q, True),
        dim=1,
        interpolation=interpolation,
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(inp, q, dim=1, interpolation=interpolation)
    _assert_output_abi(result, saved_abi, reference.shape)
    _assert_close(result, reference, torch.float32, inp.shape[1])


@pytest.mark.nanquantile_out
def test_nanquantile_noncontiguous_out():
    inp = _input((3, 1025), torch.float32)
    q = torch.tensor([0.2, 0.5, 0.8], device=inp.device)
    ref_inp, ref_q = utils.to_reference(inp, True), utils.to_reference(q, True)
    out = torch.empty((3, 3), device=inp.device).T
    out_pointer = out.data_ptr()
    ref_out = torch.empty((3, 3), dtype=ref_inp.dtype, device=ref_inp.device).T
    reference = torch.ops.aten.nanquantile.out(
        ref_inp, ref_q, 1, False, interpolation="linear", out=ref_out
    )
    result = flag_gems.nanquantile_out(inp, q, dim=1, out=out)
    assert result is out
    assert reference is ref_out
    assert out.data_ptr() == out_pointer
    assert not result.is_contiguous()
    _assert_close(result, reference, torch.float32, inp.shape[1])


@pytest.mark.nanquantile
@pytest.mark.parametrize("size", [1, 1024, 1025, 4097])
def test_nanquantile_all_nan_and_path_boundaries(size):
    inp = torch.full((3, size), float("nan"), device=flag_gems.device)
    inp[1, 0] = -float("inf")
    inp[1, -1] = float("inf")
    q = torch.tensor([0.0, 0.5, 1.0], device=inp.device)
    reference = _native_reference(
        utils.to_reference(inp, True), utils.to_reference(q, True), dim=1
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(inp, q, dim=1)
    _assert_output_abi(result, saved_abi, reference.shape)
    _assert_close(result, reference, torch.float32, size)


@pytest.mark.nanquantile
def test_nanquantile_special_values_and_scalar_tensor_q():
    inp = torch.tensor(
        [[float("nan"), -0.0, 0.0, -float("inf"), float("inf"), 2.0]],
        device=flag_gems.device,
    )
    q = torch.tensor(0.5, device=inp.device)
    reference = _native_reference(
        utils.to_reference(inp, True), utils.to_reference(q, True), dim=1
    )
    saved_abi = _input_abi(inp, q)
    result = flag_gems.nanquantile(inp, q, dim=1)
    _assert_output_abi(result, saved_abi, reference.shape)
    _assert_close(result, reference, torch.float32, inp.shape[1])
    result_valid = ~torch.isnan(result)
    reference_valid = ~torch.isnan(reference)
    utils.gems_assert_equal(
        torch.signbit(result)[result_valid],
        torch.signbit(reference)[reference_valid],
    )


@pytest.mark.nanquantile
def test_nanquantile_errors():
    inp = torch.randn((3, 5), device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(
            inp.to(torch.float16), torch.tensor(0.5, device=inp.device)
        )
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(inp, torch.ones((1, 1), device=inp.device))
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(
            inp, torch.tensor(0.5, device=inp.device), interpolation="bad"
        )
    with pytest.raises(RuntimeError):
        flag_gems.nanquantile(torch.empty(0, device=inp.device), 0.5)
