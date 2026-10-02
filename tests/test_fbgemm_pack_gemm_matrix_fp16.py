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

"""Correctness tests for ``aten::fbgemm_pack_gemm_matrix_fp16``.

Native contract measured on this target: only the ``default`` overload exists;
the weight must be float32 and rank >= 2 (rank 0/1 raise ``IndexError``, rank >= 3
packs the contiguous flat prefix ``shape[0] * shape[1]``); the result is an opaque
``cpp_custom_type_hack`` handle whose bytes are a fresh allocation per call and
therefore undefined; packing saturates to ``+-65504`` and clamps a contiguous
weight in place, while a strided weight is copied and left untouched; the handle
carries no autograd state. The operator is CPU-only FBGEMM, so both engines get
CPU tensors (a CUDA weight crashes the process instead of raising).

Because handle bytes cannot be compared, the packing format is observed only
through its documented native consumer,
``fbgemm_linear_fp16_weight_fp32_activation``.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

OP_NAME = "fbgemm_pack_gemm_matrix_fp16"

# CPU-only FBGEMM op: a CUDA weight segfaults the process instead of raising, so
# every weight and the consumer's identity/bias are built on the CPU device the
# native kernel actually supports.
_CPU = torch.device("cpu")

# Probed: the kernel reads ``data_ptr<float>()`` and rejects the other required
# dtypes (float16, bfloat16, float64, int8, uint8, int32, int64, bool,
# float8_e4m3fn, float8_e5m2) with "expected scalar type Float but found <X>".
# Those dtypes are covered by the negative rows below.
SUPPORTED_DTYPES = [torch.float32]

# The kernel reads ``size(0)``/``size(1)``, so the spec's 0-dim and 1-dim shapes
# cannot be packed; they are covered by the rank negative rows instead.
_GRID_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]

# GEMM-realistic (N, K) geometry plus zero-extent sizes; the rank >= 3 entries
# exercise the flat-prefix packing path.
_EXTRA_SHAPES = [
    (0, 7),
    (5, 0),
    (0, 0),
    (1, 1),
    (7, 17),
    (33, 65),
    (128, 256),
    (256, 512),
    (4096, 1),
    (1, 4096),
    (2, 19, 7),
    (2, 3, 4),
    (3, 5, 7, 11),
]

SHAPES = tu.selected_cases(
    _GRID_SHAPES + _EXTRA_SHAPES,
    quick=[(2, 19, 7), (7, 17), (1, 1), (0, 7), (5, 0), (0, 0)],
)


def _ramp(shape, low=-1.0, high=1.0):
    """Contiguous, coordinate-varying values in ``[low, high]``."""
    flat = torch.linspace(low, high, steps=math.prod(shape), device=_CPU)
    return flat.reshape(shape)


def _packed_readback(handle, shape):
    """Decode a packed handle with its documented native consumer.

    ``fbgemm_linear_fp16_weight_fp32_activation`` is the reader of this format;
    with an identity activation it returns the fp16-rounded weight transposed.
    """
    rows, cols = shape[0], shape[1]
    eye = torch.eye(cols, dtype=torch.float32, device=_CPU)
    bias = torch.zeros(rows, dtype=torch.float32, device=_CPU)
    return torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(eye, handle, bias)


def _assert_handle(res_out, ref_out, inp):
    """Handle metadata; its bytes are an opaque pointer and are never compared."""
    assert res_out.dtype == ref_out.dtype
    assert res_out.shape == ref_out.shape
    assert res_out.device == inp.device
    assert res_out.is_contiguous() == ref_out.is_contiguous()
    assert res_out.storage_offset() == ref_out.storage_offset()
    assert res_out.data_ptr() != inp.data_ptr()


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", SHAPES)
def test_fbgemm_pack_gemm_matrix_fp16_value_range(dtype, value_range, shape):
    # Shared value-range framework; the weight is then moved to CPU because the
    # native kernel is CPU-only.
    inp = tu.make_input(dtype, shape, value_range).cpu()
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(ref_inp)
    res_out = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)

    _assert_handle(res_out, ref_out, inp)

    # Format parity: the candidate's handle must be readable by the native
    # consumer exactly like the native handle is.
    res_packed = _packed_readback(res_out, shape)
    tu.assert_result_equal(res_packed, _packed_readback(ref_out, shape))


_SPECIAL_CASES = [
    ("nan_only", [float("nan"), 0.0, -0.0, 1.0, -1.0]),
    ("inf_only", [float("inf"), float("-inf"), 1.0, -1.0, 0.0]),
    ("nan_and_inf", [float("nan"), float("inf"), float("-inf"), 0.0, -0.0]),
    ("saturation_positive", [65504.0, 65505.0, 1e30, 3.4e38, 0.5]),
    ("saturation_negative", [-65504.0, -65505.0, -1e30, -3.4e38, -0.5]),
    ("saturation_boundary", [65503.5, -65503.5, 65504.0, -65504.0, 65505.5]),
]


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize(
    "case_name,values", tu.selected_cases(_SPECIAL_CASES, quick=[])
)
def test_fbgemm_pack_gemm_matrix_fp16_special_values(case_name, values):
    del case_name
    # Rank >= 2 is required, so the payload is packed as one row of len(values).
    inp = torch.tensor(values, dtype=torch.float32, device=_CPU).reshape(1, -1)
    shape = inp.shape
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(ref_inp)
    res_out = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)

    _assert_handle(res_out, ref_out, inp)
    # Only the native consumer defines the expected side: FBGEMM's own float to
    # fp16 step saturates inf to +-65504 and decodes nan as -65504 here, so a
    # plain "half().float()" expectation would not describe this build.
    tu.assert_result_equal(
        _packed_readback(res_out, shape),
        _packed_readback(ref_out, shape),
    )


_VIEW_CASES = tu.selected_cases(
    ["transposed", "stride_holes", "offset_slice", "expanded"],
    quick=["transposed", "stride_holes", "offset_slice", "expanded"],
)


def _view_weight(case):
    if case == "transposed":
        return _ramp((5, 8)).t()
    if case == "stride_holes":
        return _ramp((16, 20))[::2, ::3]
    if case == "offset_slice":
        return _ramp((8, 5))[1:7]
    return _ramp((1, 5)).expand(4, 5)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("case", _VIEW_CASES)
def test_fbgemm_pack_gemm_matrix_fp16_strided_weight(case):
    inp = _view_weight(case)
    ref_inp = tu.to_reference(inp)
    shape = inp.shape

    ref_out = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(ref_inp)
    res_out = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)

    _assert_handle(res_out, ref_out, inp)
    # Native packs ``weight.contiguous()``: the logical row-major values of the
    # view, with the strided base left untouched (see the saturation test).
    res_packed = _packed_readback(res_out, shape)
    tu.assert_result_equal(res_packed, _packed_readback(ref_out, shape))


_MUTATION_CASES = tu.selected_cases(
    ["contiguous", "offset_slice", "transposed", "stride_holes"],
    quick=["contiguous", "offset_slice", "transposed", "stride_holes"],
)


def _wide_weight(case):
    """Values well outside fp16 range, so in-place saturation is observable."""
    if case == "contiguous":
        return _ramp((8, 5), -1e5, 1e5)
    if case == "offset_slice":
        return _ramp((8, 5), -1e5, 1e5)[1:7]
    if case == "transposed":
        return _ramp((5, 8), -1e5, 1e5).t()
    return _ramp((16, 20), -1e5, 1e5)[::2, ::3]


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("case", _MUTATION_CASES)
def test_fbgemm_pack_gemm_matrix_fp16_inplace_saturation(case):
    inp = _wide_weight(case)
    ref_inp = tu.to_reference(inp)
    inp_ptr = inp.untyped_storage().data_ptr()

    ref_out = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(ref_inp)
    res_out = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)

    # Saturation is a side effect on the caller's tensor: native clamps a
    # contiguous weight in place (offset slices included) and copies a strided
    # weight, leaving its base untouched. Comparing the mutated inputs directly
    # makes the candidate reproduce that exact side effect, and the storage check
    # keeps an in-place operator from reallocating its input. Only the native
    # consumer defines the packed values here, because the reference weight is
    # left unclamped for the strided cases.
    tu.assert_result_equal(inp, ref_inp)
    assert inp.untyped_storage().data_ptr() == inp_ptr
    tu.assert_result_equal(
        _packed_readback(res_out, inp.shape),
        _packed_readback(ref_out, inp.shape),
    )


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("shape", [(7, 17), (33, 65)])
def test_fbgemm_pack_gemm_matrix_fp16_handle_allocation(shape):
    inp = _ramp(shape).requires_grad_(True)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(ref_inp)
    res_out = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)

    # One pointer-sized handle, never aliasing the weight storage.
    assert res_out.untyped_storage().nbytes() == ref_out.untyped_storage().nbytes()
    assert res_out.untyped_storage().data_ptr() != inp.untyped_storage().data_ptr()
    # Two packs of the same weight own independent handles; ``res_out`` is still
    # referenced here, so its address cannot have been recycled.
    again = flag_gems.fbgemm_pack_gemm_matrix_fp16(inp.clone())
    assert again.untyped_storage().data_ptr() != res_out.untyped_storage().data_ptr()
    # The format carries no autograd state on either side, so this operator has
    # no backward path (differentiating the handle with respect to the weight
    # fails with "element 0 of tensors does not require grad and does not have a
    # grad_fn").
    assert res_out.requires_grad is False
    assert res_out.grad_fn is None


# Negative rows, kept in BOTH quick and default modes. Inputs are constructed
# outside the ``pytest.raises`` context so a construction failure cannot fake a
# pass. The native operator is probed with the same call and fails for the same
# reason; only the candidate's exception is asserted. AttributeError is
# deliberately not accepted, so a missing candidate cannot pass these tests.
_UNSUPPORTED_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("dtype", _UNSUPPORTED_DTYPES)
def test_fbgemm_pack_gemm_matrix_fp16_rejects_non_float32_weight(dtype):
    # zeros (not empty) keeps this row from holding undefined data even though
    # the native kernel rejects the dtype before any read.
    inp = torch.zeros(7, 17, dtype=dtype, device=_CPU)
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("shape", [(256,), ()])
def test_fbgemm_pack_gemm_matrix_fp16_rejects_rank_below_two(shape):
    # Rank 1 is natively rejected (IndexError), so no rank-1 packing semantics
    # are fabricated anywhere in this file.
    inp = torch.zeros(shape, dtype=torch.float32, device=_CPU)
    with pytest.raises((IndexError, RuntimeError, ValueError)):
        flag_gems.fbgemm_pack_gemm_matrix_fp16(inp)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
@pytest.mark.parametrize("value", [3.14, None])
def test_fbgemm_pack_gemm_matrix_fp16_rejects_non_tensor_weight(value):
    with pytest.raises((TypeError, RuntimeError, IndexError)):
        flag_gems.fbgemm_pack_gemm_matrix_fp16(value)


@pytest.mark.fbgemm_pack_gemm_matrix_fp16
def test_fbgemm_pack_gemm_matrix_fp16_rejects_missing_weight():
    with pytest.raises((TypeError, RuntimeError)):
        flag_gems.fbgemm_pack_gemm_matrix_fp16()
