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

"""Correctness tests for ``aten::mkldnn_linear_backward_weights``.

MkldnnCPU is the only registered dispatch key: grad_output and input must be
mkldnn-layout CPU tensors and weight a dense float32 CPU tensor; the results
are dense CPU float32 tensors shaped (out_features, in_features) and
(out_features,) -- 0-dim when ``bias_defined`` is False. The oracle is
``torch.ops.aten.mkldnn_linear_backward_weights`` (``default`` and ``out``)
and the candidate is called directly as
``flag_gems.mkldnn_linear_backward_weights``.
"""

import math

import pytest
import torch

import flag_gems

from . import test_utils as tu

pytestmark = pytest.mark.mkldnn_linear_backward_weights

NATIVE = torch.ops.aten.mkldnn_linear_backward_weights
OUT_FEATURES = 8

# Probed CPU dtype support: fp32/fp16/bf16 mkldnn operands are accepted and
# always yield fp32 results. The other required dtypes are unreachable or
# rejected: to_mkldnn() only accepts float/bfloat16/half/uint8/int8, and the
# int8/uint8 path fails to build a primitive descriptor. Those rejections are
# asserted by the negative rows below instead of being silently dropped.
SUPPORTED_DTYPES = [torch.float32, torch.float16, torch.bfloat16]

# bias_defined is the operator's only parameter (bool, no schema default); both
# values stay in the quick subset as well.
BIAS_FLAGS = [True, False]

# oneDNN rejects rank 0/1 grad_output and input ("could not create a primitive
# descriptor for the reorder primitive" / "... inner product forward propagation
# primitive"), so only the rank >= 2 spec shapes apply; (1, 256) is the smallest
# valid rank-2 boundary and (2, 19, 7) keeps a quick row inside the default set.
_SPEC_SHAPES = [shape for shape in tu.selected_shapes() if len(shape) >= 2]
GRAD_SHAPES = list(
    dict.fromkeys(tu.selected_cases(_SPEC_SHAPES, quick=[]) + [(2, 19, 7), (1, 256)])
)

OUT_SHAPES = tu.selected_cases(
    [(1024, 1024), (20, 320, 15), (2, 19, 7), (1, 256)],
    quick=[(2, 19, 7), (1, 256)],
)
METADATA_SHAPES = tu.selected_cases([(20, 320, 15), (2, 19, 7)], quick=[(2, 19, 7)])

# Positive nan/inf coverage is default-only; quick selects no rows.
SPECIAL_SHAPES = tu.selected_cases([(1024, 1024), (20, 320, 15)], quick=[])
SPECIAL_CASES = tu.selected_cases(tu.special_value_cases(SUPPORTED_DTYPES), quick=[])

# Rejected argument sets, one row per probed failure mode; every row is kept in
# both modes.
NEGATIVE_CASES = [
    "dense_operands",
    "dense_input",
    "dense_grad_output",
    "weight_float64",
    "weight_bfloat16",
    "weight_float16",
    "weight_int32",
    "mkldnn_int8",
    "mkldnn_uint8",
    "rank1_grad_output",
    "rank1_input",
    "batch_mismatch",
    "empty_k_dim",
]


def _dense_observation(tensor):
    """Lossless export of an opaque mkldnn tensor for value comparison."""
    assert tensor.layout == torch._mkldnn
    return tensor.to_dense()


def _mkldnn_operands(shape, value_range, dtype):
    """Exact native operand types for the candidate and for the oracle.

    The two sets have identical values but independent storage, so a candidate
    that writes to its operands cannot affect the native reference result.
    values are built on flag_gems.device and moved to CPU because MkldnnCPU
    only accepts CPU operands; weight stays a dense float32 CPU tensor.
    """
    grad_dense = tu.make_input(dtype, (*shape[:-1], OUT_FEATURES), value_range).cpu()
    inp_dense = tu.make_input(dtype, shape, value_range).cpu()
    weight = tu.make_input(torch.float32, (OUT_FEATURES, shape[-1]), value_range).cpu()

    def mkldnn_pair(dense):
        return dense.clone().to_mkldnn(), dense.clone().to_mkldnn()

    grad_output, ref_grad_output = mkldnn_pair(grad_dense)
    inp, ref_inp = mkldnn_pair(inp_dense)
    return grad_output, inp, weight.clone(), ref_grad_output, ref_inp, weight


def _dense_operands(shape, dtype):
    """Plain dense CPU operands, used by the negative rows."""
    grad_output = tu.make_input(dtype, (*shape[:-1], OUT_FEATURES), ["-1", "1"]).cpu()
    inp = tu.make_input(dtype, shape, ["-1", "1"]).cpu()
    weight = tu.make_input(torch.float32, (OUT_FEATURES, shape[-1]), ["-1", "1"]).cpu()
    return grad_output, inp, weight


@pytest.mark.parametrize("bias_defined", BIAS_FLAGS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", GRAD_SHAPES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_linear_backward_weights(shape, value_range, bias_defined, dtype):
    grad_output, inp, weight, ref_grad_output, ref_inp, ref_weight = _mkldnn_operands(
        shape, value_range, dtype
    )

    ref_grad_weight, ref_grad_bias = NATIVE(
        ref_grad_output, ref_inp, ref_weight, bias_defined
    )
    res_grad_weight, res_grad_bias = flag_gems.mkldnn_linear_backward_weights(
        grad_output, inp, weight, bias_defined
    )

    tu.assert_result_close(res_grad_weight, ref_grad_weight)
    if bias_defined:
        tu.assert_result_close(res_grad_bias, ref_grad_bias)
    else:
        # Without a bias the native grad_bias is an undefined 0-dim
        # placeholder, so only its metadata carries information.
        assert res_grad_bias.shape == ref_grad_bias.shape
        assert res_grad_bias.dtype == ref_grad_bias.dtype


@pytest.mark.parametrize("bias_defined", BIAS_FLAGS)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("shape", OUT_SHAPES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_linear_backward_weights_out(shape, value_range, bias_defined, dtype):
    grad_output, inp, weight, ref_grad_output, ref_inp, ref_weight = _mkldnn_operands(
        shape, value_range, dtype
    )
    bias_shape = (OUT_FEATURES,) if bias_defined else ()

    ref_grad_weight = torch.zeros(OUT_FEATURES, shape[-1], dtype=torch.float32)
    ref_grad_bias = torch.zeros(bias_shape, dtype=torch.float32)
    ref_out = NATIVE.out(
        ref_grad_output,
        ref_inp,
        ref_weight,
        bias_defined,
        out0=ref_grad_weight,
        out1=ref_grad_bias,
    )

    res_grad_weight = torch.zeros(OUT_FEATURES, shape[-1], dtype=torch.float32)
    res_grad_bias = torch.zeros(bias_shape, dtype=torch.float32)
    res_out = flag_gems.mkldnn_linear_backward_weights(
        grad_output,
        inp,
        weight,
        bias_defined,
        out0=res_grad_weight,
        out1=res_grad_bias,
    )

    # The out overload returns the caller's buffers themselves.
    assert res_out[0] is res_grad_weight
    assert res_out[1] is res_grad_bias

    tu.assert_result_close(res_out[0], ref_out[0])
    if bias_defined:
        tu.assert_result_close(res_out[1], ref_out[1])
    else:
        # bias_defined=False leaves grad_bias an undefined 0-dim placeholder:
        # its contents are unspecified storage, so only metadata is asserted.
        assert res_grad_bias.shape == ref_grad_bias.shape
        assert res_grad_bias.dtype == ref_grad_bias.dtype


def _special_operands(shape, dtype, scenario):
    """mkldnn operands filled from the shared nan/inf payload.

    Repeating the payload keeps nan, inf and signed zeros exactly; regenerating
    values from a range would not.
    """
    payload = tu.make_special_input(dtype, scenario)

    def filled(numel):
        return payload.repeat(-(-numel // payload.numel()))[:numel]

    grad_shape = (*shape[:-1], OUT_FEATURES)
    grad_dense = filled(math.prod(grad_shape)).reshape(grad_shape).cpu()
    inp_dense = filled(math.prod(shape)).reshape(shape).cpu()
    weight = tu.make_input(torch.float32, (OUT_FEATURES, shape[-1]), ["-1", "1"]).cpu()

    def mkldnn_pair(dense):
        return dense.clone().to_mkldnn(), dense.clone().to_mkldnn()

    grad_output, ref_grad_output = mkldnn_pair(grad_dense)
    inp, ref_inp = mkldnn_pair(inp_dense)
    return grad_output, inp, weight.clone(), ref_grad_output, ref_inp, weight


@pytest.mark.parametrize("dtype,scenario", SPECIAL_CASES)
@pytest.mark.parametrize("shape", SPECIAL_SHAPES)
def test_mkldnn_linear_backward_weights_special_values(shape, dtype, scenario):
    grad_output, inp, weight, ref_grad_output, ref_inp, ref_weight = _special_operands(
        shape, dtype, scenario
    )

    ref_grad_weight, ref_grad_bias = NATIVE(ref_grad_output, ref_inp, ref_weight, True)
    res_grad_weight, res_grad_bias = flag_gems.mkldnn_linear_backward_weights(
        grad_output, inp, weight, True
    )

    tu.assert_result_close(res_grad_weight, ref_grad_weight)
    tu.assert_result_close(res_grad_bias, ref_grad_bias)


@pytest.mark.parametrize("bias_defined", BIAS_FLAGS)
@pytest.mark.parametrize("shape", METADATA_SHAPES)
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_mkldnn_linear_backward_weights_metadata(shape, dtype, bias_defined):
    grad_output, inp, weight, ref_grad_output, ref_inp, ref_weight = _mkldnn_operands(
        shape, ["-1", "1"], dtype
    )
    grad_before = _dense_observation(grad_output).clone()
    inp_before = _dense_observation(inp).clone()

    ref_grad_weight, ref_grad_bias = NATIVE(
        ref_grad_output, ref_inp, ref_weight, bias_defined
    )
    res_grad_weight, res_grad_bias = flag_gems.mkldnn_linear_backward_weights(
        grad_output, inp, weight, bias_defined
    )

    tu.assert_result_equal(weight, ref_weight)
    # Both mkldnn operands are read-only inputs.
    tu.assert_result_equal(_dense_observation(grad_output), grad_before)
    tu.assert_result_equal(_dense_observation(inp), inp_before)

    # Results are freshly allocated dense CPU fp32 tensors.
    assert res_grad_weight.dtype == torch.float32
    assert res_grad_weight.layout == torch.strided
    assert res_grad_weight.device == weight.device
    assert res_grad_weight.shape == (OUT_FEATURES, shape[-1])
    assert res_grad_bias.dtype == torch.float32
    assert res_grad_bias.layout == torch.strided
    assert res_grad_bias.device == weight.device
    assert res_grad_bias.shape == ref_grad_bias.shape

    tu.assert_result_close(res_grad_weight, ref_grad_weight)
    if bias_defined:
        tu.assert_result_close(res_grad_bias, ref_grad_bias)


def _negative_args(case):
    """Argument set the native operator rejects, per probed failure mode.

    Returns (grad_output, input, weight, bias_defined); each comment records
    the native error the row reproduces on this build.
    """
    if case == "dense_operands":
        # NotImplementedError: no kernel for the plain CPU backend.
        return (*_dense_operands((4, 8), torch.float32), True)
    if case == "dense_input":
        # RuntimeError: grad_output and input needs to be mkldnn layout.
        grad_output, inp, weight = _dense_operands((4, 8), torch.float32)
        return grad_output.to_mkldnn(), inp, weight, True
    if case == "dense_grad_output":
        # RuntimeError: grad_output and input needs to be mkldnn layout.
        grad_output, inp, weight = _dense_operands((4, 8), torch.float32)
        return grad_output, inp.to_mkldnn(), weight, True
    if case.startswith("weight_"):
        # RuntimeError: weight needs to be a dense float32 tensor.
        weight_dtype = getattr(torch, case.removeprefix("weight_"))
        grad_output, inp, weight = _dense_operands((4, 8), torch.float32)
        return grad_output.to_mkldnn(), inp.to_mkldnn(), weight.to(weight_dtype), True
    if case.startswith("mkldnn_"):
        # RuntimeError: no primitive descriptor for this operand dtype.
        operand_dtype = getattr(torch, case.removeprefix("mkldnn_"))
        grad_output, inp, weight = _dense_operands((4, 8), operand_dtype)
        return grad_output.to_mkldnn(), inp.to_mkldnn(), weight, True
    if case == "rank1_grad_output":
        # RuntimeError: no primitive descriptor for a 1-D grad_output.
        grad_output = tu.make_input(torch.float32, (OUT_FEATURES,), ["-1", "1"]).cpu()
        inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu().to_mkldnn()
        weight = tu.make_input(torch.float32, (OUT_FEATURES, 8), ["-1", "1"]).cpu()
        return grad_output.to_mkldnn(), inp, weight, True
    if case == "rank1_input":
        # RuntimeError: no primitive descriptor for a 1-D input.
        grad_output = (
            tu.make_input(torch.float32, (4, OUT_FEATURES), ["-1", "1"])
            .cpu()
            .to_mkldnn()
        )
        inp = tu.make_input(torch.float32, (8,), ["-1", "1"]).cpu().to_mkldnn()
        weight = tu.make_input(torch.float32, (OUT_FEATURES, 8), ["-1", "1"]).cpu()
        return grad_output, inp, weight, True
    if case == "batch_mismatch":
        # RuntimeError: the batch dimensions of grad_output and input differ.
        grad_output = (
            tu.make_input(torch.float32, (5, OUT_FEATURES), ["-1", "1"])
            .cpu()
            .to_mkldnn()
        )
        inp = tu.make_input(torch.float32, (4, 8), ["-1", "1"]).cpu().to_mkldnn()
        weight = tu.make_input(torch.float32, (OUT_FEATURES, 8), ["-1", "1"]).cpu()
        return grad_output, inp, weight, True
    if case == "empty_k_dim":
        # RuntimeError: the contraction dimension K is 0.
        grad_output = (
            tu.make_input(torch.float32, (4, 3), ["-1", "1"]).cpu().to_mkldnn()
        )
        inp = tu.make_input(torch.float32, (4, 0), ["-1", "1"]).cpu().to_mkldnn()
        weight = tu.make_input(torch.float32, (3, 0), ["-1", "1"]).cpu()
        return grad_output, inp, weight, True
    raise AssertionError(f"unknown negative case: {case}")


@pytest.mark.parametrize("case", NEGATIVE_CASES)
def test_mkldnn_linear_backward_weights_rejects_invalid_operands(case):
    grad_output, inp, weight, bias_defined = _negative_args(case)

    with pytest.raises((RuntimeError, TypeError)):
        flag_gems.mkldnn_linear_backward_weights(grad_output, inp, weight, bias_defined)
