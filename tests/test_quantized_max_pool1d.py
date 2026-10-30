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


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("shape", POOL_SHAPES)
@pytest.mark.parametrize(
    "kernel_size, stride, padding, dilation, ceil_mode", POOL_CONFIGS
)
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d(
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
    res_out = flag_gems.quantized_max_pool1d(
        res_inp,
        kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
    )
    assert res_out.device == res_inp.device

    assert res_out.dtype == in_dtype
    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    # Compare dequantized values; pool over the last dim so reduce_dim=1.
    # The native op only exists on CPU, so the reference runs on CPU and the
    # FlagGems result is moved to CPU before comparison.
    utils.gems_assert_close(
        _quantized_cpu(res_out).dequantize(),
        utils.to_reference(ref_out.dequantize(), True),
        dtype=torch.float32,
        reduce_dim=1,
    )


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_padding_neutral(in_dtype):
    # Values chosen so every window that touches padding has all its real
    # values below the qint8 floor of -128.
    if in_dtype == torch.qint32:
        scale, values = 1.0, [-200.0, -300.0, -400.0, -129.0, -500.0, -600.0]
    else:
        scale, values = 0.1, [-12.8, -12.8, -12.7, -12.8, -12.8, -12.8]
    ref_inp = torch.quantize_per_tensor(
        torch.tensor([values]), scale=scale, zero_point=0, dtype=in_dtype
    )
    res_inp = _quantized_device(ref_inp, flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 3, stride=2, padding=1)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)
    assert res_out.device == res_inp.device

    # Exact integer equality: any wrong neutral element shows up here.
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
def test_quantized_max_pool1d_all_padding_window(in_dtype):
    scale = 1.0 if in_dtype == torch.qint32 else 0.1
    ref_inp = torch.quantize_per_tensor(
        torch.tensor([[5.0]]), scale=scale, zero_point=0, dtype=in_dtype
    )
    res_inp = _quantized_device(ref_inp, flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 2, stride=1, padding=1, dilation=2)
    res_out = flag_gems.quantized_max_pool1d(
        res_inp, 2, stride=1, padding=1, dilation=2
    )
    assert res_out.device == res_inp.device

    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"kernel_size": 0}, "kernel_size must be greater than zero"),
        ({"kernel_size": -1}, "kernel_size must be greater than zero"),
        ({"kernel_size": [2, 2]}, "kernel_size must be an int"),
        ({"kernel_size": []}, "kernel_size must be an int"),
        ({"kernel_size": 2, "stride": 0}, "stride must be greater than zero"),
        ({"kernel_size": 2, "stride": [2, 2]}, "stride must be None"),
        ({"kernel_size": 2, "padding": -1}, "padding must be non-negative"),
        ({"kernel_size": 3, "padding": 2}, "padding should be at most half"),
        ({"kernel_size": 2, "padding": [0, 0]}, "padding must be an int"),
        ({"kernel_size": 2, "dilation": 0}, "dilation must be greater than zero"),
        ({"kernel_size": 2, "dilation": [1, 1]}, "dilation must be an int"),
        ({"kernel_size": 20}, "Invalid computed output size"),
    ],
)
def test_quantized_max_pool1d_invalid_params(kwargs, message):
    res_inp = _make_quantized((2, 3, 8), 0.1, 0, torch.quint8, flag_gems.device)
    ref_inp = _quantized_cpu(res_inp)

    # The same call must fail the same way on the aten reference.
    with pytest.raises(RuntimeError, match=message):
        torch.quantized_max_pool1d(ref_inp, **kwargs)
    with pytest.raises(RuntimeError, match=message):
        flag_gems.quantized_max_pool1d(res_inp, **kwargs)


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "shape, message",
    [
        ((8,), "Expected 2D or 3D input tensor"),
        ((2, 2, 2, 8), "Expected 2D or 3D input tensor"),
        ((2, 0, 8), "input dimensions must be non-zero"),
        ((2, 3, 0), "Invalid computed output size"),
    ],
)
def test_quantized_max_pool1d_invalid_shapes(shape, message):
    res_inp = _make_quantized(shape, 0.1, 0, torch.quint8, flag_gems.device)
    with pytest.raises(RuntimeError, match=message):
        torch.quantized_max_pool1d(_quantized_cpu(res_inp), 2, stride=2)
    with pytest.raises(RuntimeError, match=message):
        flag_gems.quantized_max_pool1d(res_inp, 2, stride=2)


@pytest.mark.quantized_max_pool1d
def test_quantized_max_pool1d_empty_batch():
    res_inp = _make_quantized((0, 3, 8), 0.1, 0, torch.quint8, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(_quantized_cpu(res_inp), 2, stride=2)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 2, stride=2)
    assert res_out.device == res_inp.device
    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert res_out.numel() == 0


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("layout", ["slice", "transpose", "offset"])
def test_quantized_max_pool1d_non_contiguous_input(in_dtype, layout):
    if flag_gems.vendor_name == "ascend" and layout == "transpose":
        pytest.skip("QuantizedNPU as_strided cannot construct transposed inputs")

    def _view(base):
        if layout == "slice":
            return base[:, :, ::2]
        if layout == "transpose":
            return base.transpose(1, 2)
        return base[:, :, 3:19]

    # Tensor.to("cpu") is a silent no-op for a non-contiguous quantized tensor,
    # so the reference view is built on a CPU tensor from the start.
    cpu_base = _make_quantized((2, 4, 24), 0.1, 0, in_dtype, "cpu")
    ref_inp = _view(cpu_base)
    res_inp = _view(_quantized_device(cpu_base, flag_gems.device))
    assert not res_inp.is_contiguous()
    assert res_inp.device.type != "cpu"

    ref_out = torch.quantized_max_pool1d(ref_inp, 3, stride=2, padding=1)
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)
    assert res_out.device == res_inp.device
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize(
    "in_dtype, scale, zero_point",
    [
        (torch.quint8, 0.1, 128),
        (torch.qint8, 0.1, -32),
        (torch.qint32, 1.0, 7),
    ],
)
def test_quantized_max_pool1d_quant_params(in_dtype, scale, zero_point):
    res_inp = _make_quantized((2, 3, 16), scale, zero_point, in_dtype, flag_gems.device)
    ref_out = torch.quantized_max_pool1d(
        _quantized_cpu(res_inp), 3, stride=2, padding=1
    )
    res_out = flag_gems.quantized_max_pool1d(res_inp, 3, stride=2, padding=1)
    assert res_out.device == res_inp.device

    assert res_out.q_scale() == ref_out.q_scale()
    assert res_out.q_zero_point() == ref_out.q_zero_point()
    assert torch.equal(res_out.int_repr().to("cpu"), ref_out.int_repr())


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
    return out.int_repr().reshape(n, out_l, c).permute(0, 2, 1).contiguous()


@pytest.mark.quantized_max_pool1d
@pytest.mark.parametrize("in_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("shape", [(2, 3, 1), (1, 3, 1), (3, 5, 1), (2, 1, 1)])
def test_quantized_max_pool1d_unit_length(in_dtype, shape):
    scale = 1.0
    numel = 1
    for size in shape:
        numel *= size
    fp = torch.arange(1.0, numel + 1.0).reshape(shape)
    ref_inp = torch.quantize_per_tensor(fp, scale, 0, in_dtype)
    res_inp = _quantized_device(ref_inp, flag_gems.device)

    ref_out = torch.quantized_max_pool1d(ref_inp, 2, stride=1, padding=1)

    # Pool the integer representation directly: comparing dequantized floats
    # would trip over scale round-trip error rather than the layout.
    expected = torch.nn.functional.max_pool1d(
        ref_inp.int_repr().to(torch.float32), 2, stride=1, padding=1
    )

    if not torch.equal(ref_out.int_repr().to(torch.float32), expected):
        # ATen hit the layout defect described above. Pin it so the repair below
        # cannot silently become stale: re-reading the buffer as NLC *must*
        # reproduce the independent reference exactly, otherwise this is a
        # different failure and the test should fail rather than paper over it.
        assert _aten_unit_length_layout_is_broken(ref_inp), (
            "torch.quantized_max_pool1d disagrees with max_pool1d on "
            f"{tuple(ref_inp.shape)}, which the documented ATen layout defect "
            "does not explain"
        )
        ref_int_repr = _aten_unit_length_int_repr(ref_out)
        assert torch.equal(ref_int_repr.to(torch.float32), expected)
    else:
        # Outside the affected domain the native op already agrees, so the
        # comparison above needs no repair.
        ref_int_repr = ref_out.int_repr()

    res_out = flag_gems.quantized_max_pool1d(res_inp, 2, stride=1, padding=1)
    assert res_out.device == res_inp.device

    assert tuple(res_out.shape) == tuple(ref_out.shape)
    assert torch.equal(res_out.int_repr().to("cpu"), ref_int_repr)
