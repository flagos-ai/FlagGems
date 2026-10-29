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

DTYPES = utils.FLOAT_DTYPES + [torch.int32]
TENSOR_TENSOR_SHAPES = [
    ((), ()),
    ((0,), (1,)),
    ((7,), (7,)),
    ((2, 3), (3,)),
    ((4, 1, 5), (1, 3, 1)),
]
POINTWISE_SHAPES = [(), (0,), (7,), (2, 3), (4, 3, 5)]


def _make_input(shape, dtype, *, positive=False):
    if dtype.is_floating_point:
        tensor = torch.randn(shape, dtype=dtype, device=flag_gems.device)
        return tensor.abs().add_(0.25) if positive else tensor
    return torch.randint(
        1 if positive else -3, 5, shape, dtype=dtype, device=flag_gems.device
    )


def _assert_result(result, reference):
    assert result.dtype == torch.float64
    utils.gems_assert_close(
        result, reference, torch.float64, equal_nan=True, atol=1e-12
    )


@pytest.mark.float_power_tensor_tensor
@pytest.mark.parametrize("shapes", TENSOR_TENSOR_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_tensor(shapes, dtype):
    base = _make_input(shapes[0], dtype, positive=True)
    exponent = _make_input(shapes[1], dtype)
    ref = torch.ops.aten.float_power.Tensor_Tensor(
        utils.to_reference(base), utils.to_reference(exponent)
    )

    result = flag_gems.float_power_tensor_tensor(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_tensor_tensor
def test_float_power_tensor_tensor_noncontiguous_and_special_values():
    base = torch.tensor(
        [[0.0, -0.0, 1.0, -1.0], [2.0, float("inf"), float("nan"), 4.0]],
        dtype=torch.float32,
        device=flag_gems.device,
    ).T
    exponent = torch.tensor(
        [[-1.0, 3.0, 0.5, 2.0], [0.0, -2.0, 1.0, float("inf")]],
        dtype=torch.float32,
        device=flag_gems.device,
    ).T
    ref = torch.ops.aten.float_power.Tensor_Tensor(
        utils.to_reference(base), utils.to_reference(exponent)
    )

    result = flag_gems.float_power_tensor_tensor(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_tensor_scalar
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("exponent", [2.0, -0.5])
def test_float_power_tensor_scalar(shape, dtype, exponent):
    base = _make_input(shape, dtype, positive=True)
    ref = torch.ops.aten.float_power.Tensor_Scalar(utils.to_reference(base), exponent)

    result = flag_gems.float_power_tensor_scalar(base, exponent)

    _assert_result(result, ref)


@pytest.mark.float_power_scalar_tensor
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("base", [2.0, -2.0])
def test_float_power_scalar_tensor(shape, dtype, base):
    exponent = _make_input(shape, dtype)
    ref = torch.ops.aten.float_power.Scalar(base, utils.to_reference(exponent))

    result = flag_gems.float_power_scalar_tensor(base, exponent)

    _assert_result(result, ref)


def _out_needs_cpu_reference(operator):
    vendors = {
        "float_power_scalar_tensor_out": {"kunlunxin", "iluvatar"},
        "float_power_tensor_scalar_out": {"kunlunxin", "iluvatar", "ascend", "enflame"},
        "float_power_tensor_tensor_out": {"kunlunxin", "iluvatar"},
    }
    return flag_gems.vendor_name in vendors.get(operator, ())


def _out_raw_cpu(tensor):
    # Typed device copies can narrow Double payloads on these backends.
    return torch.empty(0, dtype=tensor.dtype, device="cpu").set_(
        tensor.untyped_storage().cpu(),
        tensor.storage_offset(),
        tensor.shape,
        tensor.stride(),
    )


def _out_metadata(tensor):
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
        tensor.requires_grad,
        tensor.is_conj(),
        tensor.is_neg(),
    )


def _out_input_snapshot(tensors):
    return [
        (
            tensor,
            _out_metadata(tensor),
            _out_raw_cpu(tensor).contiguous().reshape(-1).view(torch.uint8).clone(),
        )
        for tensor in tensors
        if isinstance(tensor, torch.Tensor)
    ]


def _assert_out_inputs_unchanged(snapshots):
    for tensor, metadata, values in snapshots:
        assert _out_metadata(tensor) == metadata
        observed = _out_raw_cpu(tensor).contiguous().reshape(-1).view(torch.uint8)
        assert torch.equal(observed, values)


def _out_buffer(operator, shape):
    if _out_needs_cpu_reference(operator):
        numel = 1
        for dimension in shape:
            numel *= dimension
        return (
            torch.empty(numel * 8, dtype=torch.uint8, device=flag_gems.device)
            .view(torch.float64)
            .reshape(shape)
        )
    return torch.empty(shape, dtype=torch.float64, device=flag_gems.device)


def _out_reference(operator, *args):
    if _out_needs_cpu_reference(operator):
        # The vendor CPU float_power API is also broken on Iluvatar. Double
        # CPU pow has the same real-input promotion semantics, checked on NV.
        promoted = tuple(
            _out_raw_cpu(arg).double() if isinstance(arg, torch.Tensor) else arg
            for arg in args
        )
        return torch.pow(*promoted)
    reference_args = tuple(
        utils.to_reference(arg) if isinstance(arg, torch.Tensor) else arg
        for arg in args
    )
    tensors = [arg for arg in reference_args if isinstance(arg, torch.Tensor)]
    shape = torch.broadcast_shapes(*(tensor.shape for tensor in tensors))
    reference_out = torch.empty(shape, dtype=torch.float64, device=tensors[0].device)
    native = {
        "float_power_scalar_tensor_out": torch.ops.aten.float_power.Scalar_out,
        "float_power_tensor_scalar_out": torch.ops.aten.float_power.Tensor_Scalar_out,
        "float_power_tensor_tensor_out": torch.ops.aten.float_power.Tensor_Tensor_out,
    }[operator]
    return native(*reference_args, out=reference_out)


def _call_out_and_assert(operator, args, out, reference):
    inputs = _out_input_snapshot(args)
    original = _out_metadata(out)
    assert original[1] == torch.float64
    original_storage = _out_raw_cpu(out).untyped_storage()
    original_bytes = (
        torch.empty(0, dtype=torch.uint8, device="cpu")
        .set_(original_storage, 0, (original_storage.nbytes(),), (1,))
        .clone()
    )
    result = getattr(flag_gems, operator)(*args, out=out)
    _assert_out_inputs_unchanged(inputs)
    assert isinstance(result, torch.Tensor) and result is out
    assert result.device == original[0] and result.dtype == original[1]
    assert tuple(result.shape) == tuple(reference.shape)
    assert result.requires_grad == original[6]
    assert (result.is_conj(), result.is_neg()) == original[7:9]
    if original[2] == tuple(reference.shape):
        assert tuple(result.stride()) == original[3]
        assert result.storage_offset() == original[4]
        assert result.untyped_storage().data_ptr() == original[5]
        assert result.untyped_storage().nbytes() == original_storage.nbytes()
        # Only logical out elements may be written; retain backing padding.
        unused = torch.ones(
            original_storage.nbytes() // result.element_size(),
            dtype=torch.bool,
            device="cpu",
        )
        torch.empty(0, dtype=torch.bool, device="cpu").set_(
            unused.untyped_storage(), original[4], original[2], original[3]
        ).fill_(False)
        unused = unused.repeat_interleave(result.element_size())
        after_storage = _out_raw_cpu(out).untyped_storage()
        after_bytes = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
            after_storage, 0, (after_storage.nbytes(),), (1,)
        )
        assert torch.equal(after_bytes[unused], original_bytes[unused])
    else:
        assert tuple(result.stride()) == tuple(
            torch.empty(reference.shape, device="cpu").stride()
        )
        assert result.storage_offset() == original[4]
    if result.numel():
        assert all(
            result.untyped_storage().data_ptr() != metadata[5]
            for tensor, metadata, _ in inputs
            if tensor.numel()
        )
    # Guard the original target ABI before observing values on CPU.
    utils.gems_assert_close(
        _out_raw_cpu(result),
        _out_raw_cpu(reference),
        torch.float64,
        equal_nan=True,
        atol=1e-12,
    )
    return result


@pytest.mark.float_power_tensor_tensor_out
@pytest.mark.parametrize("shapes", TENSOR_TENSOR_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_tensor_out(shapes, dtype):
    base = _make_input(shapes[0], dtype, positive=True)
    exponent = _make_input(shapes[1], dtype)
    operator = "float_power_tensor_tensor_out"
    result_shape = torch.broadcast_shapes(*shapes)
    ref = _out_reference(operator, base, exponent)
    out = _out_buffer(operator, result_shape)

    _call_out_and_assert(operator, (base, exponent), out, ref)


@pytest.mark.float_power_tensor_scalar_out
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_tensor_scalar_out(shape, dtype):
    base = _make_input(shape, dtype, positive=True)
    operator = "float_power_tensor_scalar_out"
    ref = _out_reference(operator, base, -0.5)
    out = _out_buffer(operator, (0,))

    _call_out_and_assert(operator, (base, -0.5), out, ref)


@pytest.mark.float_power_scalar_tensor_out
@pytest.mark.parametrize("shape", POINTWISE_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_float_power_scalar_tensor_out(shape, dtype):
    exponent = _make_input(shape, dtype)
    operator = "float_power_scalar_tensor_out"
    ref = _out_reference(operator, 2.0, exponent)
    out = _out_buffer(operator, shape)

    _call_out_and_assert(operator, (2.0, exponent), out, ref)


@pytest.mark.float_power_tensor_tensor_out
def test_float_power_tensor_tensor_out_noncontiguous():
    base = torch.rand((3, 4), device=flag_gems.device)
    exponent = torch.rand((3, 4), device=flag_gems.device)
    operator = "float_power_tensor_tensor_out"
    out = _out_buffer(operator, (4, 3)).T
    ref = _out_reference(operator, base, exponent)

    result = _call_out_and_assert(operator, (base, exponent), out, ref)

    assert not result.is_contiguous()


@pytest.mark.float_power_tensor_scalar_out
def test_float_power_out_rejects_non_double_output():
    base = torch.rand(8, device=flag_gems.device)
    out = torch.empty_like(base)

    unchanged = _out_input_snapshot((base, out))
    with pytest.raises(RuntimeError, match="requires dtype Double"):
        flag_gems.float_power_tensor_scalar_out(base, 2.0, out=out)
    _assert_out_inputs_unchanged(unchanged)
