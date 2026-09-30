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

"""Correctness tests for fbgemm_linear_fp16_weight_fp32_activation.

Host FBGEMM path: ``packed_weight`` is the opaque handle built by
``fbgemm_pack_gemm_matrix_fp16`` from a host weight of shape
(out_features=N, in_features=K). The activation must be float32 with rank >= 2
and the result is ``activation.shape[:-1] + (N,)`` in float32.
"""

import pytest
import torch

import flag_gems

from . import test_utils as tu

# Host-only operator: the reference and the candidate both receive these CPU
# operands, including the very same opaque packed-weight handle.
_HOST = "cpu"

ACTIVATION_DTYPE = torch.float32

# A non-float32 activation is rejected natively ("expected scalar type Float but
# found ..."), so the dtype grid rides on the bias. An fp8 bias is rejected by
# native promotion ("Promotion for Float8 Types is not supported") and a complex
# bias by the cast back to the float32 result, which leaves these nine valid
# bias dtypes.
BIAS_DTYPES = [
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
    torch.bool,
]

# Output columns of the packed weight.
N = 32

# Activation shapes; the last dim is K. Rank 0 and rank 1 cannot be expressed by
# this operator at all and are covered by the negative rank rows, so the grid
# keeps only the ranks it accepts. The quick row is a member of the default list.
SHAPE_ROWS = [
    ((1, 1), 1),
    ((0, 7), N),
    ((2, 19, 7), N),
    ((1024, 1024), N),
    ((20, 320, 15), N),
    ((16, 128, 64, 60), N),
    ((16, 7, 57, 32, 29), N),
]
QUICK_SHAPE_ROWS = SHAPE_ROWS[:3]

_PACKED = {}


def _packed_weight(k, n, value_range):
    """Pack the host weight (out_features, in_features) = (n, k).

    The packer stores fp16 and saturates anything outside that range (it emits
    "FOUND weight out of range"), which is its defined behaviour, so the full
    fp32 value ranges are used unchanged and a saturated operand stays valid and
    reproducible.
    """
    key = ("range", k, n, tuple(value_range))
    if key not in _PACKED:
        weight = tu.make_input(torch.float32, (n, k), value_range).to(_HOST)
        _PACKED[key] = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    return _PACKED[key]


def _constant_packed_weight(k, n, fill):
    """Pack a constant host weight; used by the special-value rows."""
    key = ("const", k, n, fill)
    if key not in _PACKED:
        weight = torch.full((n, k), fill, dtype=torch.float32, device=_HOST)
        _PACKED[key] = torch.ops.aten.fbgemm_pack_gemm_matrix_fp16(weight)
    return _PACKED[key]


def _make_input(dtype, shape, value_range):
    return tu.make_input(dtype, shape, value_range).to(_HOST)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("dtype", BIAS_DTYPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize(
    "shape,n", tu.selected_cases(SHAPE_ROWS, quick=QUICK_SHAPE_ROWS)
)
def test_fbgemm_linear_fp16_weight_fp32_activation(shape, n, value_range, dtype):
    inp = _make_input(ACTIVATION_DTYPE, shape, value_range)
    bias = _make_input(dtype, (n,), value_range)
    packed = _packed_weight(shape[-1], n, value_range)

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    assert res_out.shape == tuple(shape[:-1]) + (n,)
    assert res_out.device == inp.device
    tu.assert_result_close(res_out, ref_out)


# A length-1 bias is broadcast over every leading dimension of a 3-dim and a
# 5-dim activation. Quick keeps both dtype branches on the cheap shape; the
# default suite adds the large spec shapes on top of them.
BROADCAST_ROWS = [
    ((2, 19, 7), N, torch.bfloat16),
    ((2, 19, 7), N, torch.int32),
    ((20, 320, 15), N, torch.bfloat16),
    ((16, 7, 57, 32, 29), N, torch.int32),
]
QUICK_BROADCAST_ROWS = BROADCAST_ROWS[:2]


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize(
    "shape,n,dtype", tu.selected_cases(BROADCAST_ROWS, quick=QUICK_BROADCAST_ROWS)
)
def test_fbgemm_linear_fp16_weight_fp32_activation_broadcast(shape, n, dtype):
    inp = _make_input(ACTIVATION_DTYPE, shape, ["-1", "1"])
    bias = _make_input(dtype, (1,), ["-1", "1"])
    packed = _packed_weight(shape[-1], n, ["-1", "1"])

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    tu.assert_result_close(res_out, ref_out)


# Non-contiguous activations with a fixed K: a transposed batch pair and an
# inner-dimension slice. Both are accepted natively and cheap enough for either
# level, so they are not mode-gated.
LAYOUT_ROWS = [
    ("transposed", (4, 8, 16)),
    ("sliced", (4, 32)),
]


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("layout,base_shape", LAYOUT_ROWS)
def test_fbgemm_linear_fp16_weight_fp32_activation_non_contiguous(layout, base_shape):
    base = _make_input(ACTIVATION_DTYPE, base_shape, ["-1", "1"])
    if layout == "transposed":
        # (8, 4, 16) with the batch pair swapped, K = 16 unchanged.
        inp = base.transpose(0, 1)
    else:
        # (4, 16) with an inner stride of 2, K = 16 unchanged.
        inp = base[:, ::2]
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    packed = _packed_weight(inp.shape[-1], N, ["-1", "1"])

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_empty_output():
    # N = 0: a zero-element float32 result with a zero-length bias.
    inp = _make_input(ACTIVATION_DTYPE, (4, 16), ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (0,), ["-1", "1"])
    packed = _packed_weight(16, 0, ["-1", "1"])

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    assert res_out.device == inp.device
    tu.assert_result_equal(res_out, ref_out)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_zero_k_metadata():
    # Native value gap for K=0: QuantizedLinear.cpp allocates at::empty, while
    # FBGEMM's cblas_gemm_compute writes only inside its k_ind < k loop.
    # The subsequent bias add therefore reads uninitialized storage. This case
    # checks metadata only, not numerical correctness of the empty reduction.
    # PyTorch 5228986c39: aten/src/ATen/native/QuantizedLinear.cpp;
    # FBGEMM dbc3157bf256f1339b3fa1fef2be89ac4078be0e:
    # include/fbgemm/FbgemmFPCommon.h, cblas_gemm_compute.
    inp = _make_input(ACTIVATION_DTYPE, (4, 0), ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    packed = _packed_weight(0, N, ["-1", "1"])

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    assert res_out.shape == ref_out.shape
    assert res_out.dtype == ref_out.dtype
    assert res_out.device == ref_out.device


# float32 is this operator's only activation dtype, so the activation-side
# special-value matrix is that dtype x {nan, inf, mixed}; the extra row packs an
# all-nan weight. Positive special inputs are default-only.
SPECIAL_K = 8
SPECIAL_ROWS = tu.selected_cases(
    [(scenario, False) for _, scenario in tu.special_value_cases([ACTIVATION_DTYPE])]
    + [("nan", True)],
    quick=[],
)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("scenario,nan_weight", SPECIAL_ROWS)
def test_fbgemm_linear_fp16_weight_fp32_activation_special_values(scenario, nan_weight):
    payload = tu.make_special_input(ACTIVATION_DTYPE, scenario).to(_HOST)
    # Constant rows keep the arithmetic well defined: an all-inf row times a
    # uniform positive weight is exactly +-inf, not a 0 * inf nan.
    inp = payload.reshape(payload.numel(), 1).repeat(1, SPECIAL_K)
    bias = torch.zeros(N, dtype=ACTIVATION_DTYPE, device=_HOST)
    packed = _constant_packed_weight(SPECIAL_K, N, float("nan") if nan_weight else 0.5)

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    tu.assert_result_close(res_out, ref_out)


# Bias-side special values for every floating dtype the bias natively accepts.
# The integer and bool bias dtypes cannot carry nan/inf at all.
BIAS_SPECIAL_ROWS = tu.selected_cases(
    tu.special_value_cases(
        [torch.float16, torch.bfloat16, torch.float32, torch.float64]
    ),
    quick=[],
)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("dtype,scenario", BIAS_SPECIAL_ROWS)
def test_fbgemm_linear_fp16_weight_fp32_activation_special_bias(dtype, scenario):
    # A nan/inf bias propagates straight through the trailing add.
    payload = tu.make_special_input(dtype, scenario).to(_HOST)
    # Tile the 5-value payload up to the (n,) bias length.
    bias = payload.repeat(-(-N // payload.numel()))[:N]
    inp = _make_input(ACTIVATION_DTYPE, (4, SPECIAL_K), ["-1", "1"])
    packed = _constant_packed_weight(SPECIAL_K, N, 0.5)

    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, tu.to_reference(bias)
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    tu.assert_result_close(res_out, ref_out)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_readonly_operands():
    inp = _make_input(ACTIVATION_DTYPE, (4, 16), ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    packed = _packed_weight(16, N, ["-1", "1"])
    inp_snapshot = inp.clone()
    bias_snapshot = bias.clone()
    # The packed weight is opaque, so its state is observed by replaying the
    # native op on a fixed probe activation.
    probe = _make_input(ACTIVATION_DTYPE, (2, 16), ["-1", "1"])
    handle_snapshot = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        probe, packed, bias_snapshot
    )

    flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)

    tu.assert_result_equal(inp, inp_snapshot)
    tu.assert_result_equal(bias, bias_snapshot)
    tu.assert_result_equal(
        torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
            probe, packed, bias_snapshot
        ),
        handle_snapshot,
    )


# The native op differentiates only through its trailing bias add: the packed
# activation never reaches the autograd graph, while d(out)/d(bias) is an
# AddBackward0 reduction. Backward workloads are default-only.
BACKWARD_ROWS = tu.selected_cases([(4, 16), (2, 3, 16)], quick=[])


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("shape", BACKWARD_ROWS)
def test_fbgemm_linear_fp16_weight_fp32_activation_backward_bias(shape):
    inp = _make_input(ACTIVATION_DTYPE, shape, ["-1", "1"])
    upstream = _make_input(ACTIVATION_DTYPE, tuple(shape[:-1]) + (N,), ["-1", "1"])
    # Differentiate through the original bias leaf.
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"]).requires_grad_(True)
    packed = _packed_weight(shape[-1], N, ["-1", "1"])

    ref_bias = tu.to_reference(bias)
    ref_out = torch.ops.aten.fbgemm_linear_fp16_weight_fp32_activation(
        tu.to_reference(inp), packed, ref_bias
    )
    res_out = flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)
    tu.assert_result_close(res_out, ref_out)

    ref_grad = torch.autograd.grad(
        ref_out, ref_bias, grad_outputs=tu.to_reference(upstream)
    )[0]
    res_grad = torch.autograd.grad(res_out, bias, grad_outputs=upstream)[0]

    assert res_grad.shape == (N,)
    tu.assert_result_close(res_grad, ref_grad)


# The native op rejects the arguments below; the candidate must reject them too.
# Each operand is built before the pytest.raises block so that a construction
# failure cannot be mistaken for a rejected call.


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("shape", [(), (N,)])
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_rank(shape):
    inp = _make_input(ACTIVATION_DTYPE, shape, ["-1", "1"])
    packed = _packed_weight(N, N, ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    with pytest.raises((IndexError, RuntimeError)):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)


UNSUPPORTED_ACTIVATION_DTYPES = [
    torch.float64,
    torch.float16,
    torch.bfloat16,
    torch.int8,
    torch.uint8,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("dtype", UNSUPPORTED_ACTIVATION_DTYPES)
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_activation_dtype(dtype):
    inp = _make_input(dtype, (4, 16), ["-1", "1"])
    packed = _packed_weight(16, N, ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_complex_bias(dtype):
    # The trailing add has no float32 result for a complex bias: "result type
    # ComplexFloat can't be cast to the desired output type Float".
    inp = _make_input(ACTIVATION_DTYPE, (4, 16), ["-1", "1"])
    bias = torch.ones(N, dtype=dtype, device=_HOST)
    packed = _packed_weight(16, N, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)


NEGATIVE_BIAS_SHAPES = [(1, N), (), (1, 1, N), (N + 1,), (0,)]


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
@pytest.mark.parametrize("bias_shape", NEGATIVE_BIAS_SHAPES)
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_bias_shape(bias_shape):
    inp = _make_input(ACTIVATION_DTYPE, (4, 16), ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, bias_shape, ["-1", "1"])
    packed = _packed_weight(16, N, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_k_mismatch():
    inp = _make_input(ACTIVATION_DTYPE, (4, 15), ["-1", "1"])
    packed = _packed_weight(16, N, ["-1", "1"])
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, packed, bias)


@pytest.mark.fbgemm_linear_fp16_weight_fp32_activation
def test_fbgemm_linear_fp16_weight_fp32_activation_negative_weight_type():
    # The packed-weight argument is an opaque cpp-type wrapper, never a dense
    # tensor of bytes.
    inp = _make_input(ACTIVATION_DTYPE, (4, 16), ["-1", "1"])
    raw = torch.zeros(16 * N, dtype=torch.uint8, device=_HOST)
    bias = _make_input(ACTIVATION_DTYPE, (N,), ["-1", "1"])
    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.fbgemm_linear_fp16_weight_fp32_activation(inp, raw, bias)
