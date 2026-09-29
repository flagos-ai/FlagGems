# Copyright 2026 FlagOS Contributors.
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

# quantized_batch_norm operates on a 4D (NCHW) quantized tensor and produces a
# quantized tensor with caller-supplied output_scale / output_zero_point. The
# reference aten kernel is only available on the CPU quantized backend, so the
# reference path always runs on CPU regardless of the ``--ref`` flag.

# Representative NCHW shapes covering small/medium/large batch and spatial sizes.
QBN_SHAPES = [
    (1, 3, 4, 4),
    (2, 3, 4, 4),
    (4, 8, 16, 16),
    (2, 16, 32, 32),
    (8, 3, 64, 64),
    (1, 64, 128, 128),
]

QBN_QUANT_DTYPES = [torch.quint8, torch.qint8]

# (output_scale, output_zero_point) pairs exercising a range of output
# quantization parameters, including values that clamp to the limits. The high
# zero_point is pinned to 127 (the qint8 maximum) rather than 128 so the same
# parametrization is valid for both quint8 and qint8: the public
# ``torch.quantize_per_tensor`` used to pre-allocate the ``out`` tensor validates
# the zero_point against the dtype range, and 128 is out of range for qint8.
QBN_OUT_PARAMS = [
    (0.1, 0),
    (0.1, 127),
    (0.01, 10),
    (0.25, 3),
]

# (input_scale, input_zero_point) pairs for the input quantized tensor.
QBN_IN_PARAMS = [
    (0.5, 0),
    (0.05, 10),
    (1.0, 3),
]

# Zero-sized shapes: ATen returns an empty clone preserving the input's
# quantization parameters, for every degenerate dimension.
QBN_EMPTY_SHAPES = [
    (0, 3, 4, 4),  # N == 0
    (2, 0, 4, 4),  # C == 0
    (2, 3, 0, 4),  # H == 0
    (2, 3, 4, 0),  # W == 0
]


def _make_quantized_input(shape, scale, zero_point, dtype, device):
    fp = torch.randn(shape, device="cpu")
    return torch.quantize_per_tensor(fp, scale, zero_point, dtype).to(device)


def _make_channels_last_input(shape, scale, zero_point, dtype, device):
    """Build a channels-last per-tensor quantized tensor from fixed integers.

    The integer representation is drawn from a seeded generator so CPU and GPU
    paths share identical values; the tensor is materialized in the channels-last
    layout through ``quantize_per_tensor`` on a channels-last float tensor (which
    ATen preserves). A CPU quantized tensor cannot simply be moved to the device:
    ``.to()`` normalises the strides to contiguous.
    """
    C = shape[1]
    spatial = shape[2] * shape[3]
    gen = torch.Generator().manual_seed(42)
    ints = torch.randint(0, 256, (shape[0], C, spatial), generator=gen).to(torch.uint8)
    dequant = (ints.float() - zero_point) * scale
    return torch.quantize_per_tensor(
        dequant.reshape(shape).to(memory_format=torch.channels_last).to(device),
        scale,
        zero_point,
        dtype,
    )


def _qbn_params(C, device):
    weight = torch.randn(C, dtype=torch.float32, device="cpu")
    bias = torch.randn(C, dtype=torch.float32, device="cpu")
    mean = torch.randn(C, dtype=torch.float32, device="cpu")
    var = torch.rand(C, dtype=torch.float32, device="cpu") + 0.5
    return weight, bias, mean, var


def _make_quantized_from_int(int_tensor, scale, zero_point, dtype):
    """Reconstruct a per-tensor quantized tensor that shares ``int_tensor``'s
    exact integer representation, using only the public ``quantize_per_tensor``
    API (the private ``torch._make_per_tensor_quantized_tensor`` is forbidden by
    Rule 8). Dequantize the target integers to their float values, then
    re-quantize: the round-trip is exact for already-quantized integers.
    """
    return torch.quantize_per_tensor(
        (int_tensor.float() - zero_point) * scale, scale, zero_point, dtype
    )


def _assert_quant_equal(res, ref):
    """Compare two quantized tensors by their integer representations.

    quantized_batch_norm produces a fixed-precision quantized tensor whose
    values are an exact function of its inputs (round-to-even + clamp), so we
    compare the integer representation byte-for-byte rather than the
    dequantized float values (which would require tolerance).
    """
    res_int = res.int_repr()
    ref_int = ref.int_repr()
    if res_int.device != ref_int.device:
        res_int = res_int.to(ref_int.device)
    assert (
        res_int.dtype == ref_int.dtype
    ), f"int_repr dtype mismatch: {res_int.dtype} vs {ref_int.dtype}"
    assert (
        res_int.shape == ref_int.shape
    ), f"int_repr shape mismatch: {res_int.shape} vs {ref_int.shape}"
    utils.gems_assert_equal(res_int, ref_int)
    # Quantization parameters should also match.
    assert (
        abs(res.q_scale() - ref.q_scale()) < 1e-6
    ), f"output scale mismatch: {res.q_scale()} vs {ref.q_scale()}"
    assert (
        res.q_zero_point() == ref.q_zero_point()
    ), f"output zero_point mismatch: {res.q_zero_point()} vs {ref.q_zero_point()}"


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("shape", QBN_SHAPES)
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("in_params", QBN_IN_PARAMS)
@pytest.mark.parametrize("out_params", QBN_OUT_PARAMS)
def test_quantized_batch_norm(shape, in_dtype, in_params, out_params):
    in_scale, in_zero_point = in_params
    out_scale, out_zero_point = out_params
    C = shape[1]

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    # Rebuild the reference input to share the integer representation with the
    # GPU input so both paths see identical values.
    res_int = res_qx.int_repr().to("cpu")
    ref_qx = _make_quantized_from_int(res_int, in_scale, in_zero_point, in_dtype)

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    res_out = flag_gems.quantized_batch_norm(
        res_qx, weight, bias, mean, var, 1e-5, out_scale, out_zero_point
    )

    _assert_quant_equal(res_out, ref_out)


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("in_params", QBN_IN_PARAMS[:1])
@pytest.mark.parametrize("out_params", QBN_OUT_PARAMS[:2])
def test_quantized_batch_norm_channels_last(in_dtype, in_params, out_params):
    """A channels-last input must produce the same values as ATen and keep the
    channels-last layout on the output (the kernel must not write into a
    detached, reallocated buffer)."""
    out_scale, out_zero_point = out_params
    in_scale, in_zero_point = in_params
    shape = (2, 3, 4, 4)
    C = shape[1]

    # ``_make_channels_last_input`` draws its integers from a fixed seed, so the
    # CPU reference and the device input share identical values *and* the
    # channels-last layout.
    res_qx = _make_channels_last_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    assert res_qx.is_contiguous(memory_format=torch.channels_last)
    ref_qx = _make_channels_last_input(shape, in_scale, in_zero_point, in_dtype, "cpu")
    assert ref_qx.is_contiguous(memory_format=torch.channels_last)
    utils.gems_assert_equal(res_qx.int_repr().cpu(), ref_qx.int_repr())

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    res_out = flag_gems.quantized_batch_norm(
        res_qx, weight, bias, mean, var, 1e-5, out_scale, out_zero_point
    )

    _assert_quant_equal(res_out, ref_out)
    # ATen keeps a channels-last input in the channels-last layout.
    assert res_out.is_contiguous(
        memory_format=torch.channels_last
    ), "channels-last input should produce a channels-last output"
    assert ref_out.is_contiguous(memory_format=torch.channels_last)


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("shape", QBN_EMPTY_SHAPES)
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
def test_quantized_batch_norm_empty(shape, in_dtype):
    """numel == 0: ATen returns an empty clone preserving the input qparams."""
    C = shape[1]

    res_qx = _make_quantized_input(shape, 0.5, 10, in_dtype, flag_gems.device)
    ref_qx = _make_quantized_input(shape, 0.5, 10, in_dtype, "cpu")
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, 0.1, 3
    )
    res_out = flag_gems.quantized_batch_norm(
        res_qx, weight, bias, mean, var, 1e-5, 0.1, 3
    )

    assert res_out.numel() == 0
    assert res_out.dtype == ref_out.dtype
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    # The empty result must preserve the *input's* quantization parameters,
    # not the requested output ones.
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("in_scale", [0.0, 0.1])
@pytest.mark.parametrize("out_scale,out_zero_point", [(0.0, 0), (0.0, 10), (1e-30, 3)])
def test_quantized_batch_norm_zero_output_scale(
    in_dtype, in_scale, out_scale, out_zero_point
):
    """output_scale=0.0 must degrade like native, not crash on the host.

    The fused parameters involve ``input_scale / output_scale`` and
    ``(bias - inner) / output_scale``; C++ double division by zero yields
    inf/NaN that the requantization clamp folds to a defined result, so ATen
    computes normally. A host-side Python division would raise
    ZeroDivisionError before the kernel launches.
    """
    shape = (2, 3, 4, 4)
    C = shape[1]
    res_qx = _make_quantized_input(shape, in_scale, 3, in_dtype, flag_gems.device)
    ref_qx = _make_quantized_input(shape, in_scale, 3, in_dtype, "cpu")
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)

    ref_out = torch.quantized_batch_norm(
        ref_qx,
        weight.to("cpu"),
        bias.to("cpu"),
        mean.to("cpu"),
        var.to("cpu"),
        1e-5,
        out_scale,
        out_zero_point,
    )
    res_out = flag_gems.quantized_batch_norm(
        res_qx, weight, bias, mean, var, 1e-5, out_scale, out_zero_point
    )

    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    utils.gems_assert_equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("param", ["weight", "bias", "mean", "var"])
@pytest.mark.parametrize("bad_size", [1, 2, 4])
def test_quantized_batch_norm_bad_size(bad_size, param):
    shape = (2, 3, 4, 4)
    C = shape[1]
    res_qx = _make_quantized_input(shape, 0.5, 10, torch.quint8, flag_gems.device)
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    params = {"weight": weight, "bias": bias, "mean": mean, "var": var}
    params[param] = torch.randn(bad_size, device=flag_gems.device)

    with pytest.raises(RuntimeError):
        flag_gems.quantized_batch_norm(
            res_qx, **params, eps=1e-5, output_scale=0.1, output_zero_point=3
        )


@pytest.mark.quantized_batch_norm
@pytest.mark.parametrize("param", ["weight", "bias", "mean", "var"])
@pytest.mark.parametrize("bad_dtype", [torch.float64, torch.int32, torch.uint8])
def test_quantized_batch_norm_bad_dtype(bad_dtype, param):
    shape = (2, 3, 4, 4)
    C = shape[1]
    res_qx = _make_quantized_input(shape, 0.5, 10, torch.quint8, flag_gems.device)
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    params = {"weight": weight, "bias": bias, "mean": mean, "var": var}
    params[param] = torch.randn(C, device=flag_gems.device).to(bad_dtype)

    with pytest.raises(RuntimeError):
        flag_gems.quantized_batch_norm(
            res_qx, **params, eps=1e-5, output_scale=0.1, output_zero_point=3
        )


@pytest.mark.quantized_batch_norm_out
@pytest.mark.parametrize("shape", QBN_SHAPES)
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
@pytest.mark.parametrize("out_params", QBN_OUT_PARAMS)
def test_quantized_batch_norm_out(shape, in_dtype, out_params):
    out_scale, out_zero_point = out_params
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    res_int = res_qx.int_repr().to("cpu")
    ref_qx = _make_quantized_from_int(res_int, in_scale, in_zero_point, in_dtype)

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    # Pre-allocate the out tensor on GPU with the target output quantization params.
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        out_scale,
        out_zero_point,
        in_dtype,
    )

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        out_scale,
        out_zero_point,
        out=res_out,
    )

    _assert_quant_equal(res_out, ref_out)


@pytest.mark.quantized_batch_norm_out
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
def test_quantized_batch_norm_out_mismatched_qparams(in_dtype):
    """The out tensor's stale scale/zero_point must be replaced by the ones the
    call requests (ATen updates the output qparams)."""
    shape = (2, 3, 4, 4)
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0
    out_scale, out_zero_point = 0.1, 3

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    ref_qx = _make_quantized_from_int(
        res_qx.int_repr().to("cpu"), in_scale, in_zero_point, in_dtype
    )

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    # Stale quantization parameters (0.9 / 100) that differ from the requested
    # output parameters (0.1 / 3).
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        0.9,
        100,
        in_dtype,
    )

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        out_scale,
        out_zero_point,
        out=res_out,
    )

    _assert_quant_equal(res_out, ref_out)
    assert abs(res_out.q_scale() - out_scale) < 1e-9
    assert res_out.q_zero_point() == out_zero_point


@pytest.mark.quantized_batch_norm_out
def test_quantized_batch_norm_out_resizes():
    """A ``out`` tensor with a mismatched shape is resized, ATen-style."""
    shape = (2, 3, 4, 4)
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0
    out_scale, out_zero_point = 0.1, 3

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, torch.quint8, flag_gems.device
    )
    ref_qx = _make_quantized_from_int(
        res_qx.int_repr().to("cpu"), in_scale, in_zero_point, torch.quint8
    )

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    res_out = torch.quantize_per_tensor(
        torch.zeros((2, 3, 8, 8), dtype=torch.float32, device=flag_gems.device),
        0.9,
        100,
        torch.quint8,
    )

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        out_scale,
        out_zero_point,
        out=res_out,
    )

    assert tuple(res_out.shape) == shape
    _assert_quant_equal(res_out, ref_out)


@pytest.mark.quantized_batch_norm_out
@pytest.mark.parametrize("in_dtype", QBN_QUANT_DTYPES)
def test_quantized_batch_norm_out_storage_offset(in_dtype):
    """A sliced ``out`` (non-zero storage offset) must be written in the right
    region of its backing storage, and leave neighbouring elements untouched."""
    shape = (2, 3, 4, 4)
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0
    out_scale, out_zero_point = 0.1, 3

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, in_dtype, flag_gems.device
    )
    ref_qx = _make_quantized_from_int(
        res_qx.int_repr().to("cpu"), in_scale, in_zero_point, in_dtype
    )

    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, out_scale, out_zero_point
    )

    # Backing storage of 4 batches; the out region is batches 1..2.
    res_backing = torch.quantize_per_tensor(
        torch.zeros((4, 3, 4, 4), dtype=torch.float32, device=flag_gems.device),
        0.9,
        100,
        in_dtype,
    )
    res_full = res_backing.reshape(4, 3, 4, 4)
    # Snapshot the regions outside the slice before the call.
    outside_before = torch.cat(
        [res_full[0].int_repr().reshape(-1), res_full[3].int_repr().reshape(-1)]
    ).clone()
    res_sliced = res_full[1:3]
    assert res_sliced.storage_offset() != 0

    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        out_scale,
        out_zero_point,
        out=res_sliced,
    )

    # The slice carries the updated qparams, values land inside the slice only.
    _assert_quant_equal(res_sliced, ref_out)
    assert res_sliced.storage_offset() != 0
    # Regions outside the slice are untouched (`quantize_per_tensor` of a zeros
    # tensor stores zero_point in the integer representation, i.e. 100 here).
    outside_after = torch.cat(
        [res_full[0].int_repr().reshape(-1), res_full[3].int_repr().reshape(-1)]
    )
    # Both sides are device tensors produced by this test (no CPU reference is
    # involved), so compare them directly: utils.gems_assert_equal routes
    # through to_cpu(), which under --ref=cpu asserts that the *reference*
    # lives on the host and would fail here for the wrong reason.
    torch.testing.assert_close(outside_after, outside_before, atol=0, rtol=0)


@pytest.mark.quantized_batch_norm_out
def test_quantized_batch_norm_out_empty():
    """numel == 0 through the out overload: ATen resizes ``out`` to the input's
    shape and copies the empty clone (which carries the input's qparams) in."""
    shape = (0, 3, 4, 4)
    C = shape[1]
    in_scale, in_zero_point = 0.5, 0

    res_qx = _make_quantized_input(
        shape, in_scale, in_zero_point, torch.quint8, flag_gems.device
    )
    ref_qx = _make_quantized_input(shape, in_scale, in_zero_point, torch.quint8, "cpu")
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    ref_weight = weight.to("cpu")
    ref_bias = bias.to("cpu")
    ref_mean = mean.to("cpu")
    ref_var = var.to("cpu")

    ref_out = torch.quantized_batch_norm(
        ref_qx, ref_weight, ref_bias, ref_mean, ref_var, 1e-5, 0.1, 3
    )

    res_out = torch.quantize_per_tensor(
        torch.zeros((2, 3, 4, 4), dtype=torch.float32, device=flag_gems.device),
        0.9,
        100,
        torch.quint8,
    )
    flag_gems.quantized_batch_norm_out(
        res_qx,
        weight,
        bias,
        mean,
        var,
        1e-5,
        0.1,
        3,
        out=res_out,
    )

    assert tuple(res_out.shape) == shape
    assert res_out.numel() == 0
    assert abs(res_out.q_scale() - ref_out.q_scale()) < 1e-9
    assert res_out.q_zero_point() == ref_out.q_zero_point()


@pytest.mark.quantized_batch_norm_out
def test_quantized_batch_norm_out_wrong_dtype():
    shape = (2, 3, 4, 4)
    C = shape[1]
    res_qx = _make_quantized_input(shape, 0.5, 0, torch.quint8, flag_gems.device)
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device=flag_gems.device),
        0.1,
        3,
        torch.qint8,
    )
    with pytest.raises(RuntimeError, match="Expected out tensor to have dtype"):
        flag_gems.quantized_batch_norm_out(
            res_qx, weight, bias, mean, var, 1e-5, 0.1, 3, out=res_out
        )


@pytest.mark.quantized_batch_norm_out
def test_quantized_batch_norm_out_wrong_device():
    shape = (2, 3, 4, 4)
    C = shape[1]
    res_qx = _make_quantized_input(shape, 0.5, 0, torch.quint8, flag_gems.device)
    weight, bias, mean, var = _qbn_params(C, flag_gems.device)
    res_out = torch.quantize_per_tensor(
        torch.zeros(shape, dtype=torch.float32, device="cpu"), 0.1, 3, torch.quint8
    )
    with pytest.raises(RuntimeError, match="Expected out tensor to have device"):
        flag_gems.quantized_batch_norm_out(
            res_qx, weight, bias, mean, var, 1e-5, 0.1, 3, out=res_out
        )
