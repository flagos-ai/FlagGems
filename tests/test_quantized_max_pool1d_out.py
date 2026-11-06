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

QUANT_DTYPES = [torch.quint8, torch.qint8, torch.qint32]

POOL_SHAPES = [
    (4, 16),  # 2D input (N, L)
    (2, 3, 16),  # 3D input (N, C, L)
    (1, 8),  # minimal 2D
    (1, 1, 8),  # minimal 3D
    (32, 50257),  # large reduce dim, sequence-like
    (8, 3, 8192),  # large reduce dim, channel-like
]

POOL_CONFIGS = [
    (2, 2, 0, 1, False),
    (3, 2, 1, 1, False),
    (3, 2, 1, 1, True),  # ceil_mode
    (2, 1, 0, 1, False),  # stride=1, no padding
    (2, 1, 0, 2, False),  # dilation
    (5, 3, 2, 1, False),  # larger kernel
    (3, 2, 1, 1, True),  # ceil_mode + padding
]


def _quantized_device(value, device):
    if flag_gems.vendor_name == "ascend" and torch.device(device).type != "cpu":
        result = torch._empty_affine_quantized(
            value.shape,
            scale=value.q_scale(),
            zero_point=value.q_zero_point(),
            dtype=value.dtype,
            device=device,
        )
        result.untyped_storage().copy_(value.untyped_storage())
        return result
    return value.to(device)


def _quantized_cpu(value):
    if flag_gems.vendor_name == "ascend" and value.device.type != "cpu":
        int_dtype = {
            torch.quint8: torch.uint8,
            torch.qint8: torch.int8,
            torch.qint32: torch.int32,
        }[value.dtype]
        ints = torch.empty(0, dtype=int_dtype).set_(
            value.untyped_storage().cpu(),
            value.storage_offset(),
            value.shape,
            value.stride(),
        )
        return torch._make_per_tensor_quantized_tensor(
            ints, value.q_scale(), value.q_zero_point()
        )
    return value.to("cpu")


def _make_quantized(shape, scale, zero_point, dtype, device):
    torch.manual_seed(0)
    fp = torch.randn(shape, device="cpu").clamp_(-2, 2)
    return _quantized_device(
        torch.quantize_per_tensor(fp, scale=scale, zero_point=zero_point, dtype=dtype),
        device,
    )


def _integer_values(value):
    # Observe the original quantized storage through an integer view. This also
    # handles strided outputs without requiring backend quantized copy kernels.
    integer_dtype = {
        torch.quint8: torch.uint8,
        torch.qint8: torch.int8,
        torch.qint32: torch.int32,
    }[value.dtype]
    view = torch.empty(0, dtype=integer_dtype, device=value.device).set_(
        value.untyped_storage(), value.storage_offset(), value.shape, value.stride()
    )
    return view.to("cpu")


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out(
    shape, kernel_size, stride, padding, dilation, ceil_mode, in_dtype
):
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_inp = _quantized_cpu(res_inp)

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )

    scale = float(res_inp.q_scale())
    zero_point = int(res_inp.q_zero_point())
    out_shape = ref_out.shape
    res_out = _quantized_device(
        torch.quantize_per_tensor(
            torch.zeros(out_shape), scale=scale, zero_point=zero_point, dtype=in_dtype
        ),
        flag_gems.device,
    )

    got = flag_gems.quantized_max_pool1d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        out=res_out,
    )
    assert got.device == res_inp.device

    # The .out variant must return the same tensor that was passed in.
    assert got is res_out
    assert got.data_ptr() == res_out.data_ptr()
    assert got.dtype == in_dtype
    utils.gems_assert_close(
        _quantized_cpu(res_out).dequantize(),
        utils.to_reference(ref_out.dequantize(), True),
        dtype=torch.float32,
        reduce_dim=1,
    )


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_non_contiguous(in_dtype):
    """A strided out must be filled in place without touching its neighbours."""
    if flag_gems.vendor_name == "ascend":
        pytest.skip(
            "QuantizedNPU _copy_from cannot update output quantization parameters"
        )
    shape, out_len = (2, 3, 16), 8
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(
        _quantized_cpu(res_inp), 3, stride=2, padding=1
    )
    assert ref_out.shape[-1] == out_len

    # A view with stride 2 along the last dim: the odd lanes must survive.
    buffer = _make_quantized(
        shape[:-1] + (out_len * 2,), 0.5, 0, in_dtype, flag_gems.device
    )
    out = buffer[..., ::2]
    pointer_before = out.data_ptr()
    stride_before = out.stride()
    offset_before = out.storage_offset()
    untouched_before = _integer_values(buffer)[..., 1::2].to("cpu").clone()

    got = flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)
    assert got.device == res_inp.device

    assert got is out
    assert got.data_ptr() == pointer_before
    assert got.stride() == stride_before
    assert got.storage_offset() == offset_before
    assert torch.equal(_integer_values(out).to("cpu"), _integer_values(ref_out))
    assert torch.equal(_integer_values(buffer)[..., 1::2].to("cpu"), untouched_before)


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_storage_offset(in_dtype):
    """An out with a nonzero storage offset must be written at that offset."""
    if flag_gems.vendor_name == "ascend":
        pytest.skip(
            "QuantizedNPU _copy_from cannot update output quantization parameters"
        )
    shape, out_len, head = (2, 3, 16), 8, 4
    res_inp = _make_quantized(shape, 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(
        _quantized_cpu(res_inp), 3, stride=2, padding=1
    )

    buffer = _make_quantized(
        shape[:-1] + (out_len + 2 * head,), 0.5, 0, in_dtype, flag_gems.device
    )
    out = buffer[..., head : head + out_len]
    pointer_before = out.data_ptr()
    stride_before = out.stride()
    offset_before = out.storage_offset()
    before = _integer_values(buffer).to("cpu").clone()

    got = flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)
    assert got.device == res_inp.device

    assert got is out
    assert got.data_ptr() == pointer_before
    assert got.stride() == stride_before
    assert got.storage_offset() == offset_before
    assert torch.equal(_integer_values(out).to("cpu"), _integer_values(ref_out))
    assert torch.equal(
        _integer_values(buffer)[..., :head].to("cpu"), before[..., :head]
    )
    assert torch.equal(
        _integer_values(buffer)[..., head + out_len :].to("cpu"),
        before[..., head + out_len :],
    )


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_out_resizes_and_adopts_quant_params(in_dtype):
    """A wrongly-shaped out is resized and takes the input's quant params."""
    if flag_gems.vendor_name == "ascend":
        pytest.skip("QuantizedNPU cannot resize quantized output storage metadata")
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(
        _quantized_cpu(res_inp), 3, stride=2, padding=1
    )

    # Wrong shape and deliberately different scale/zero_point.
    out = _quantized_device(
        torch.quantize_per_tensor(
            torch.zeros(1, 1, 1), scale=0.5, zero_point=1, dtype=in_dtype
        ),
        flag_gems.device,
    )

    got = flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=out)
    assert got.device == res_inp.device

    assert tuple(got.shape) == tuple(ref_out.shape)
    assert got.q_scale() == ref_out.q_scale()
    assert got.q_zero_point() == ref_out.q_zero_point()
    assert torch.equal(_integer_values(got).to("cpu"), _integer_values(ref_out))


@pytest.mark.quantized_max_pool1d_out
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize(
    "shape, kernel_size, stride, padding, dilation",
    [
        ((1, 8), 2, 2, 0, 1),  # the reviewer's case: 8 -> 4, shrinks
        ((2, 3, 8), 2, 2, 0, 1),  # 3D shrink
        ((1, 8), 2, 1, 1, 1),  # 8 -> 9, grows and reallocates
    ],
)
def test_quantized_max_pool1d_out_is_input(
    in_dtype, shape, kernel_size, stride, padding, dilation
):
    """``out=input`` must pool the *original* input, not the resized one."""
    if flag_gems.vendor_name == "ascend":
        pytest.skip("QuantizedNPU cannot resize quantized output storage metadata")
    # Deterministic data derived from the case, not from a global RNG, so the
    # values a failure prints are reproducible.
    seed = sum(shape) * 1000 + kernel_size * 100 + stride * 10 + padding
    gen = torch.Generator().manual_seed(seed)
    fp = torch.randint(0, 100, shape, generator=gen).to(torch.float32)
    ref_inp = torch.quantize_per_tensor(fp, 1.0, 0, in_dtype)
    res_inp = _quantized_device(ref_inp, flag_gems.device)

    ref_out = torch.quantized_max_pool1d(
        ref_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
    )
    # Guard the test itself: the resize must actually change the shape, or the
    # ordering this covers would never be exercised.
    assert tuple(ref_out.shape) != tuple(ref_inp.shape)

    got = flag_gems.quantized_max_pool1d_out(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        out=res_inp,
    )
    assert got.device == res_inp.device

    # The result is written through the tensor that was passed in.
    assert got.data_ptr() == res_inp.data_ptr()
    assert tuple(got.shape) == tuple(ref_out.shape)
    assert got.q_scale() == ref_out.q_scale()
    assert got.q_zero_point() == ref_out.q_zero_point()
    # Integer equality: a truncated or zeroed input cannot pass this.
    assert torch.equal(_integer_values(got).to("cpu"), _integer_values(ref_out))


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out_rejects_mismatched_out():
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, torch.quint8, flag_gems.device)
    bad_dtype = _quantized_device(
        torch.quantize_per_tensor(
            torch.zeros(2, 3, 8), scale=0.1, zero_point=0, dtype=torch.qint8
        ),
        flag_gems.device,
    )
    with pytest.raises(RuntimeError, match="dtype"):
        flag_gems.quantized_max_pool1d_out(
            res_inp, 3, stride=2, padding=1, out=bad_dtype
        )


@pytest.mark.quantized_max_pool1d_out
def test_quantized_max_pool1d_out_rejects_wrong_device():
    res_inp = _make_quantized((2, 3, 16), 0.1, 0, torch.quint8, flag_gems.device)
    cpu_out = torch.quantize_per_tensor(
        torch.zeros(2, 3, 8), scale=0.1, zero_point=0, dtype=torch.quint8
    )
    with pytest.raises(RuntimeError, match="device"):
        flag_gems.quantized_max_pool1d_out(res_inp, 3, stride=2, padding=1, out=cpu_out)


def _aten_unit_length_layout_is_broken(q_input):
    """The documented domain of the ATen layout defect described below.

    Used only to attribute an observed mismatch: if ATen and ``max_pool1d``
    disagree outside this domain, something else changed and the test must not
    silently repair it.
    """
    return q_input.dim() == 3 and q_input.shape[1] > 1 and q_input.shape[-1] == 1


def _aten_unit_length_int_repr(out):
    """The integer result ATen's mislabelled buffer actually holds.

    See ``test_quantized_max_pool1d_unit_length`` for the full description. The
    bytes are correct; only the strides are wrong, so reading the raw buffer as
    (N, L, C) and permuting back to (N, C, L) recovers the intended result.
    """
    n, c = out.shape[0], out.shape[1]
    out_l = out.shape[-1]
    return _integer_values(out).reshape(n, out_l, c).permute(0, 2, 1).contiguous()
