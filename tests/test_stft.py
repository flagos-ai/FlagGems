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

import json

import pytest
import torch

import flag_gems

from . import accuracy_utils as utils

pytestmark = pytest.mark.stft

COMMON_DTYPES = (torch.float32, torch.complex64)
NVIDIA_DTYPES = (torch.float16, torch.float64, torch.complex32, torch.complex128)
DTYPES = COMMON_DTYPES + (NVIDIA_DTYPES if flag_gems.vendor_name == "nvidia" else ())


def test_stft_unsupported_shell_input_dtype():
    for name in (
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e5m2fnuz",
        "float4_e2m1fn_x2",
    ):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            inp = torch.empty(16, dtype=dtype)
            with pytest.raises(NotImplementedError):
                flag_gems.stft(inp, 8, window=torch.ones(8), return_complex=True)


@pytest.mark.parametrize("amplitude", [0.0, 1e-30, 1e20])
def test_stft_finite_extremes(amplitude):
    for dtype in COMMON_DTYPES:
        inp_cpu = _make_input((2, 1025), dtype) * amplitude
        inp = inp_cpu.to(flag_gems.device)
        for n_fft in (7, 256):
            window_cpu = torch.hann_window(n_fft)
            kwargs = dict(hop_length=3, normalized=True, return_complex=True)
            expected = torch.ops.aten.stft.default(
                inp_cpu.to(_reference_dtype(dtype)),
                n_fft,
                window=window_cpu.double(),
                **kwargs,
            )
            actual = flag_gems.stft(
                inp, n_fft, window=window_cpu.to(flag_gems.device), **kwargs
            )
            assert torch.isfinite(actual.cpu()).all()
            # Rescale on the CPU so ordinary absolute tolerances cannot hide
            # flushing tiny inputs to zero or overflow in an error reduction.
            scale = amplitude if amplitude else 1.0
            torch.testing.assert_close(
                actual.cpu().to(torch.complex128) / scale,
                expected / scale,
                rtol=1e-4,
                atol=1e-4,
            )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stft_window_promotes_low_precision_input(dtype):
    inp_cpu = _make_input((2, 1025), torch.float32).to(dtype)
    window_cpu = torch.hann_window(127, dtype=torch.float32)
    kwargs = dict(hop_length=31, win_length=127, return_complex=True)
    expected = torch.ops.aten.stft.default(
        inp_cpu.double(), 257, window=window_cpu.double(), **kwargs
    )
    actual = flag_gems.stft(
        inp_cpu.to(flag_gems.device),
        257,
        window=window_cpu.to(flag_gems.device),
        **kwargs,
    )
    assert actual.dtype == torch.complex64
    _assert_result(actual, expected.to(torch.complex64), 257)


@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="Ascend fused FFT storage-promotion regression",
)
@pytest.mark.parametrize(
    "input_dtype,window_dtype",
    [(torch.float32, torch.complex32), (torch.complex32, torch.float32)],
)
def test_stft_ascend_complex_half_promotion(input_dtype, window_dtype):
    # The effective FFT dtype is complex64, but complex-half operands still
    # require half-precision component pointers while gathering their frames.
    inp_cpu = _make_input((2, 641), input_dtype)
    window_cpu = _make_input((256,), window_dtype)
    kwargs = dict(hop_length=64, normalized=True, onesided=False, return_complex=True)
    expected = torch.ops.aten.stft.default(
        inp_cpu.to(_reference_dtype(input_dtype)),
        256,
        window=window_cpu.to(_reference_dtype(window_dtype)),
        **kwargs,
    )
    actual = flag_gems.stft(
        inp_cpu.to(flag_gems.device),
        256,
        window=window_cpu.to(flag_gems.device),
        **kwargs,
    )
    assert actual.dtype == torch.complex64
    _assert_result(actual, expected.to(torch.complex64), 256)


# A fixed set exercises the semantics without multiplying every option together.
# Half/complex-half cases use only power-of-two transforms.
CASES = (
    pytest.param((65,), 1, 1, None, "none", False, None, True, None, id="fft1"),
    pytest.param((19,), 3, 1, 3, "ones", False, False, True, None, id="fft3"),
    pytest.param((2, 31), 7, 2, 5, "hann", True, None, True, None, id="fft7-short-odd"),
    pytest.param(
        (2, 65), 16, None, 8, "hamming", False, True, True, None, id="fft16-short-even"
    ),
    pytest.param((531,), 257, 73, 129, "hann", False, False, True, None, id="fft257"),
    pytest.param(
        (2, 815), 400, 160, 399, "ones", True, None, True, None, id="fft400-short-odd"
    ),
    pytest.param(
        (2, 1100),
        512,
        128,
        512,
        "none",
        False,
        True,
        True,
        False,
        id="fft512-align-false",
    ),
    pytest.param(
        (2200,),
        1024,
        256,
        512,
        "hann",
        True,
        False,
        True,
        True,
        id="fft1024-align-true",
    ),
    pytest.param(
        (2, 4100), 2048, 512, 1024, "hamming", False, None, True, None, id="fft2048"
    ),
)

CENTER_CASES = (
    pytest.param((31,), 7, 2, 5, "hann", "reflect", None, True, None, id="reflect-odd"),
    pytest.param(
        (2, 65), 16, 4, 8, "ones", "constant", True, True, None, id="constant-even"
    ),
    pytest.param(
        (2, 257),
        16,
        5,
        7,
        "hamming",
        "replicate",
        False,
        True,
        None,
        id="replicate-odd",
    ),
    pytest.param(
        (100,), 16, 4, 16, "none", "circular", None, True, None, id="circular"
    ),
)


def _make_input(shape, dtype, noncontiguous=False):
    torch.manual_seed(23)
    if dtype.is_complex:
        part = torch.float64 if dtype == torch.complex128 else torch.float32
        value = torch.complex(
            torch.randn(shape, dtype=part), torch.randn(shape, dtype=part)
        )
    else:
        value = torch.randn(
            shape, dtype=torch.float64 if dtype == torch.float64 else torch.float32
        )
    value = value.to(dtype)
    if noncontiguous:
        value = torch.stack((value, value), dim=-1)[..., 0]
        assert not value.is_contiguous()
    return value


def _device_copy(value):
    if value is None:
        return None
    if value.is_contiguous():
        return value.to(flag_gems.device)
    # Tensor.to() can compact a sliced tensor. Construct the slice on device
    # so the test actually exercises noncontiguous kernel input/window strides.
    result = torch.stack((value, value), dim=-1).to(flag_gems.device)[..., 0]
    assert not result.is_contiguous()
    return result


def _make_window(kind, length, dtype, noncontiguous=False):
    if kind == "none":
        return None
    real_dtype = (
        torch.float64
        if dtype in (torch.float64, torch.complex128)
        else (
            torch.float16
            if dtype in (torch.float16, torch.complex32)
            else torch.float32
        )
    )
    construction_dtype = torch.float32 if real_dtype == torch.float16 else real_dtype
    if kind == "ones":
        value = torch.ones(length, dtype=real_dtype)
    elif kind == "hann":
        value = torch.hann_window(length, periodic=True, dtype=construction_dtype).to(
            real_dtype
        )
    elif kind == "hamming":
        value = torch.hamming_window(
            length, periodic=False, dtype=construction_dtype
        ).to(real_dtype)
    elif kind == "complex":
        real = torch.hann_window(length, dtype=construction_dtype).to(real_dtype)
        value = torch.complex(real, real * 0.25)
    else:
        raise ValueError(kind)
    if noncontiguous:
        value = torch.stack((value, value), dim=-1)[:, 0]
        assert not value.is_contiguous()
    return value


def _reference_dtype(dtype):
    return torch.complex128 if dtype.is_complex else torch.float64


def _assert_result(actual, expected, n_fft):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    # The high-precision FFT oracle stays on CPU in both reference modes.
    utils.gems_assert_close(
        actual.cpu(), expected, actual.dtype, reduce_dim=n_fft.bit_length()
    )


@pytest.mark.stft
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,n_fft,hop,win_length,window_kind,normalized,onesided,return_complex,align",
    CASES,
)
def test_stft(
    shape,
    n_fft,
    hop,
    win_length,
    window_kind,
    normalized,
    onesided,
    return_complex,
    align,
    dtype,
):
    if dtype in (torch.float16, torch.complex32) and n_fft & (n_fft - 1):
        pytest.skip("half precision STFT requires a power-of-two n_fft")
    inp_cpu = _make_input(shape, dtype, noncontiguous=n_fft == 16)
    window_cpu = _make_window(
        window_kind, win_length or n_fft, dtype, noncontiguous=n_fft == 16
    )
    if dtype.is_complex:
        onesided = False
    kwargs = dict(
        hop_length=hop,
        win_length=win_length,
        normalized=normalized,
        onesided=onesided,
        return_complex=return_complex,
        align_to_window=align,
    )
    ref = torch.ops.aten.stft.default(
        inp_cpu.to(_reference_dtype(dtype)),
        n_fft,
        window=(
            None
            if window_cpu is None
            else window_cpu.to(_reference_dtype(window_cpu.dtype))
        ),
        **kwargs,
    )
    inp = _device_copy(inp_cpu)
    window = _device_copy(window_cpu)
    actual = flag_gems.stft(inp, n_fft, window=window, **kwargs)
    expected_dtype = (
        torch.complex32
        if dtype in (torch.float16, torch.complex32)
        else (
            torch.complex128
            if dtype in (torch.float64, torch.complex128)
            else torch.complex64
        )
    )
    assert actual.dtype == expected_dtype
    _assert_result(actual, ref.to(expected_dtype), n_fft)


@pytest.mark.stft_center
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "shape,n_fft,hop,win_length,window_kind,pad_mode,onesided,return_complex,align",
    CENTER_CASES,
)
def test_stft_center(
    shape,
    n_fft,
    hop,
    win_length,
    window_kind,
    pad_mode,
    onesided,
    return_complex,
    align,
    dtype,
):
    if dtype in (torch.float16, torch.complex32) and n_fft & (n_fft - 1):
        pytest.skip("half precision STFT requires a power-of-two n_fft")
    inp_cpu = _make_input(shape, dtype, noncontiguous=pad_mode == "constant")
    window_cpu = _make_window(
        window_kind, win_length, dtype, noncontiguous=pad_mode == "constant"
    )
    if dtype.is_complex:
        onesided = False
    kwargs = dict(
        hop_length=hop,
        win_length=win_length,
        center=True,
        pad_mode=pad_mode,
        normalized=False,
        onesided=onesided,
        return_complex=return_complex,
        align_to_window=align,
    )
    ref = torch.ops.aten.stft.center(
        inp_cpu.to(_reference_dtype(dtype)),
        n_fft,
        window=(
            None
            if window_cpu is None
            else window_cpu.to(_reference_dtype(window_cpu.dtype))
        ),
        **kwargs,
    )
    inp = _device_copy(inp_cpu)
    window = _device_copy(window_cpu)
    actual = flag_gems.stft_center(inp, n_fft, window=window, **kwargs)
    expected_dtype = (
        torch.complex32
        if dtype in (torch.float16, torch.complex32)
        else (
            torch.complex128
            if dtype in (torch.float64, torch.complex128)
            else torch.complex64
        )
    )
    assert actual.dtype == expected_dtype
    _assert_result(actual, ref.to(expected_dtype), n_fft)


@pytest.mark.stft
@pytest.mark.parametrize(
    "entry,center",
    [
        ("python", False),
        ("python", True),
        ("default", False),
        ("center", False),
        ("center", True),
    ],
)
def test_stft_torch_dispatch(entry, center, tmp_path):
    inp_cpu = _make_input((2, 65), torch.float32)
    window_cpu = torch.hann_window(8)
    ref_inp = inp_cpu.double().requires_grad_()
    ref_window = window_cpu.double().requires_grad_()
    expected = torch.stft(
        ref_inp,
        16,
        hop_length=4,
        win_length=8,
        window=ref_window,
        center=center,
        return_complex=True,
    )
    cotangent = torch.randn(expected.shape, dtype=torch.complex64)
    expected_grads = torch.autograd.grad(
        expected, (ref_inp, ref_window), cotangent.to(expected.dtype)
    )
    inp = inp_cpu.to(flag_gems.device).requires_grad_()
    window = window_cpu.to(flag_gems.device).requires_grad_()
    log_path = tmp_path / "stft-dispatch.log"
    library = torch.library.Library("aten", "IMPL")
    previous_registrar = flag_gems.current_work_registrar
    try:
        flag_gems.only_enable(
            lib=library,
            include=["stft", "stft_center"],
            registrar=flag_gems.GeneralOpRegistrar,
            record=True,
            path=log_path,
        )
        op = torch.stft if entry == "python" else getattr(torch.ops.aten.stft, entry)
        kwargs = {} if entry == "default" else {"center": center}
        actual = op(
            inp,
            16,
            hop_length=4,
            win_length=8,
            window=window,
            return_complex=True,
            **kwargs,
        )
        grads = torch.autograd.grad(actual, (inp, window), cotangent.to(actual.device))
        # Device registrations must not replace CPU composite/autograd kernels.
        cpu_actual = op(
            ref_inp,
            16,
            hop_length=4,
            win_length=8,
            window=ref_window,
            return_complex=True,
            **kwargs,
        )
        torch.testing.assert_close(cpu_actual, expected)
    finally:
        library._destroy()
        flag_gems.current_work_registrar = previous_registrar
    _assert_result(actual, expected.to(torch.complex64), 16)
    for actual_grad, expected_grad in zip(grads, expected_grads):
        utils.gems_assert_close(
            actual_grad.cpu(), expected_grad, actual_grad.dtype, reduce_dim=16
        )
    assert "STFT" in log_path.read_text().upper()


@pytest.mark.stft
@pytest.mark.parametrize("center", [False, True])
def test_stft_real_output_and_complex_window_promotion(center):
    inp_cpu = torch.randn(2, 65)
    real_window_cpu = torch.hann_window(8)
    complex_window_cpu = torch.complex(real_window_cpu, real_window_cpu * 0.25)
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    gems_op = flag_gems.stft_center if center else flag_gems.stft
    extra = {"center": True, "pad_mode": "constant"} if center else {}
    for window_cpu, return_complex, onesided in (
        (real_window_cpu, False, True),
        (complex_window_cpu, True, False),
    ):
        kwargs = dict(
            hop_length=4,
            win_length=8,
            return_complex=return_complex,
            onesided=onesided,
            **extra,
        )
        ref = ref_op(
            inp_cpu.double(),
            16,
            window=window_cpu.to(
                torch.complex128 if window_cpu.is_complex() else torch.float64
            ),
            **kwargs,
        )
        actual = gems_op(
            inp_cpu.to(flag_gems.device),
            16,
            window=window_cpu.to(flag_gems.device),
            **kwargs,
        )
        assert actual.dtype == (torch.complex64 if return_complex else torch.float32)
        _assert_result(actual, ref.to(actual.dtype), 16)


@pytest.mark.stft
@pytest.mark.parametrize(
    "center,align", [(False, None), (False, False), (False, True), (True, None)]
)
@pytest.mark.parametrize("complex_input", [False, True])
def test_stft_gradients(center, align, complex_input):
    dtype = torch.complex64 if complex_input else torch.float32
    inp_cpu = _make_input((2, 65), dtype)
    window_cpu = _make_window("complex" if complex_input else "hann", 8, dtype)
    ref_inp = inp_cpu.to(_reference_dtype(dtype)).detach().requires_grad_()
    ref_window = (
        window_cpu.to(_reference_dtype(window_cpu.dtype)).detach().requires_grad_()
    )
    kwargs = dict(
        hop_length=4,
        win_length=8,
        window=ref_window,
        return_complex=True,
        align_to_window=align,
    )
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    if center:
        kwargs.update(center=True, pad_mode="constant")
    ref_out = ref_op(ref_inp, 16, **kwargs)
    # Supply the same cotangent explicitly: vendor complex abs backward is
    # independent of STFT and is not a reliable gradient generator on all devices.
    grad_cpu = torch.randn(ref_out.shape, dtype=torch.complex64)
    ref_grads = torch.autograd.grad(
        ref_out, (ref_inp, ref_window), grad_outputs=grad_cpu.to(ref_out.dtype)
    )

    inp = inp_cpu.to(flag_gems.device).detach().requires_grad_()
    window = window_cpu.to(flag_gems.device).detach().requires_grad_()
    kwargs["window"] = window
    op = flag_gems.stft_center if center else flag_gems.stft
    out = op(inp, 16, **kwargs)
    grads = torch.autograd.grad(
        out, (inp, window), grad_outputs=grad_cpu.to(device=out.device, dtype=out.dtype)
    )
    for actual, expected in zip(grads, ref_grads):
        assert actual.shape == expected.shape
        utils.gems_assert_close(actual.cpu(), expected, actual.dtype, reduce_dim=16)


@pytest.mark.stft_center
@pytest.mark.parametrize(
    "pad_mode,hop,complex_input,complex_window,onesided",
    [
        ("reflect", 4, False, False, False),
        ("replicate", 16, True, False, False),
        ("circular", 20, False, True, False),
        ("constant", 4, False, False, True),
    ],
)
def test_stft_padding_and_promotion_gradients(
    pad_mode, hop, complex_input, complex_window, onesided
):
    inp_cpu = _make_input((2, 65), torch.complex64 if complex_input else torch.float32)
    window_cpu = _make_window("complex" if complex_window else "hann", 8, torch.float32)
    ref_inp = inp_cpu.to(_reference_dtype(inp_cpu.dtype)).requires_grad_()
    ref_window = window_cpu.to(_reference_dtype(window_cpu.dtype)).requires_grad_()
    kwargs = dict(
        hop_length=hop,
        win_length=8,
        center=True,
        pad_mode=pad_mode,
        normalized=True,
        onesided=onesided,
        return_complex=True,
    )
    expected = torch.ops.aten.stft.center(ref_inp, 16, window=ref_window, **kwargs)
    cotangent = torch.randn(expected.shape, dtype=torch.complex64)
    expected_grads = torch.autograd.grad(
        expected, (ref_inp, ref_window), cotangent.to(expected.dtype)
    )
    inp = inp_cpu.to(flag_gems.device).requires_grad_()
    window = window_cpu.to(flag_gems.device).requires_grad_()
    actual = flag_gems.stft_center(inp, 16, window=window, **kwargs)
    _assert_result(actual, expected.to(torch.complex64), 16)
    actual_grads = torch.autograd.grad(
        actual, (inp, window), cotangent.to(actual.device)
    )
    for grad, ref_grad in zip(actual_grads, expected_grads):
        utils.gems_assert_close(grad.cpu(), ref_grad, grad.dtype, reduce_dim=16)


@pytest.mark.stft_center
def test_stft_empty_signal_gradients():
    inp = torch.empty((2, 0), device=flag_gems.device, requires_grad=True)
    window = torch.ones(8, device=flag_gems.device, requires_grad=True)
    actual = flag_gems.stft_center(
        inp, 8, window=window, center=True, pad_mode="constant", return_complex=True
    )
    cotangent = torch.ones(actual.shape, dtype=torch.complex64).to(actual.device)
    dx, dw = torch.autograd.grad(actual, (inp, window), cotangent)
    assert dx.shape == inp.shape and dx.numel() == 0
    torch.testing.assert_close(dw.cpu(), torch.zeros(8), atol=0, rtol=0)


@pytest.mark.stft
@pytest.mark.parametrize("view", ["conjugate", "negative"])
def test_stft_lazy_views(view):
    xbase = _make_input((65,), torch.complex64).to(flag_gems.device)
    wbase = _make_window("complex", 8, torch.float32).to(flag_gems.device)
    x = xbase.conj() if view == "conjugate" else xbase.conj().imag
    w = wbase.conj() if view == "conjugate" else wbase.conj().imag
    assert x.is_conj() if view == "conjugate" else x.is_neg()
    x.requires_grad_()
    w.requires_grad_()
    rx = x.detach().cpu().to(_reference_dtype(x.dtype)).requires_grad_()
    rw = w.detach().cpu().to(_reference_dtype(w.dtype)).requires_grad_()
    kwargs = dict(hop_length=4, win_length=8, return_complex=True)
    expected = torch.ops.aten.stft.default(rx, 16, window=rw, **kwargs)
    actual = flag_gems.stft(x, 16, window=w, **kwargs)
    _assert_result(actual, expected.to(torch.complex64), 16)
    # A negative cotangent view also needs its metadata honored by the VJP.
    cotangent_cpu = torch.randn(actual.shape, dtype=torch.complex64)
    cotangent = torch._neg_view(cotangent_cpu.to(actual.device))
    grads = torch.autograd.grad(actual, (x, w), cotangent)
    refs = torch.autograd.grad(expected, (rx, rw), -cotangent_cpu.to(expected.dtype))
    for grad, ref in zip(grads, refs):
        utils.gems_assert_close(grad.cpu(), ref, grad.dtype, reduce_dim=16)


@pytest.mark.stft
def test_stft_chunked_forward_and_backward(monkeypatch):
    import importlib

    module = importlib.import_module("flag_gems.ops.stft")
    monkeypatch.setattr(module, "_FRAME_BYTES", 1024)
    inp_cpu = _make_input((2, 129), torch.complex64)
    window_cpu = _make_window("complex", 16, torch.float32)
    ref_inp = inp_cpu.to(torch.complex128).requires_grad_()
    ref_window = window_cpu.to(torch.complex128).requires_grad_()
    kwargs = dict(hop_length=7, win_length=16, normalized=True, return_complex=True)
    expected = torch.ops.aten.stft.default(ref_inp, 32, window=ref_window, **kwargs)
    cotangent = torch.randn(expected.shape, dtype=torch.complex64)
    ref_grads = torch.autograd.grad(
        expected, (ref_inp, ref_window), cotangent.to(expected.dtype)
    )
    inp = inp_cpu.to(flag_gems.device).requires_grad_()
    window = window_cpu.to(flag_gems.device).requires_grad_()
    actual = flag_gems.stft(inp, 32, window=window, **kwargs)
    _assert_result(actual, expected.to(torch.complex64), 32)
    grads = torch.autograd.grad(actual, (inp, window), cotangent.to(actual.device))
    for grad, ref_grad in zip(grads, ref_grads):
        utils.gems_assert_close(grad.cpu(), ref_grad, grad.dtype, reduce_dim=32)


def _check_stft_nonfinite(center, kind, n_fft, win_length, record_property):
    inp_cpu = torch.ones(4 * n_fft, dtype=torch.float32)
    inp_cpu[13] = float(kind)
    hop = n_fft // 2
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    kwargs = {"center": True, "pad_mode": "constant"} if center else {}
    kwargs.update(hop_length=hop, win_length=win_length, return_complex=True)
    inp = inp_cpu.to(flag_gems.device)
    # Exceptional-value masks can differ with FFT butterfly ordering. Check
    # finite frames and propagation separately; retain native mask differences
    # as test-report metadata, including when the native reference uses CPU FFT.
    reference = ref_op(inp_cpu, n_fft, **kwargs)
    expected = reference if utils.TO_CPU else ref_op(inp, n_fft, **kwargs).cpu()
    actual = (flag_gems.stft_center if center else flag_gems.stft)(inp, n_fft, **kwargs)
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    # Use the same canonical ATen layout oracle as the finite-value tests.
    # A vendor's native CPU-fallback copy can make its output contiguous.
    assert actual.stride() == reference.stride()
    assert actual.device == inp.device
    actual_cpu = actual.cpu()
    starts = torch.arange(actual.shape[-1]) * hop - (n_fft // 2 if center else 0)
    affected = (starts <= 13) & (13 < starts + n_fft)
    assert torch.isfinite(actual_cpu[:, ~affected]).all()
    assert not torch.isfinite(actual_cpu[:, affected]).any()
    utils.gems_assert_close(
        actual_cpu[:, ~affected], reference[:, ~affected], actual.dtype, reduce_dim=3
    )
    comparison = {
        "reference_device": "cpu" if utils.TO_CPU else str(inp.device),
        "n_fft": n_fft,
        "win_length": n_fft if win_length is None else win_length,
        "contaminated_input_index": 13,
        "gems_stride": actual.stride(),
        "native_stride": expected.stride(),
    }
    for name, value in (("gems", actual_cpu), ("native", expected)):
        components = torch.view_as_real(value)
        comparison[name] = {
            "nan": torch.isnan(components).nonzero().tolist(),
            "posinf": torch.isposinf(components).nonzero().tolist(),
            "neginf": torch.isneginf(components).nonzero().tolist(),
        }
    comparison["masks_equal"] = comparison["gems"] == comparison["native"]
    record_property("nonfinite_native_comparison", json.dumps(comparison))


@pytest.mark.stft
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("kind", ["nan", "inf", "-inf"])
def test_stft_nonfinite(center, kind, record_property):
    _check_stft_nonfinite(center, kind, 8, None, record_property)


@pytest.mark.stft
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("kind", ["nan", "inf", "-inf"])
@pytest.mark.parametrize("win_length", [256, 127])
def test_stft_nonfinite_window_support(center, kind, win_length, record_property):
    # Index 13 lies outside the short window in the frame starting at zero.
    # Multiplication by a zero window weight must still propagate NaN/Inf.
    _check_stft_nonfinite(center, kind, 256, win_length, record_property)


@pytest.mark.stft
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("shape", [(0,), (2, 0)])
def test_stft_empty_batch_or_signal_follows_native(center, shape):
    inp_cpu = torch.empty(shape)
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    kwargs = {"center": True, "pad_mode": "constant"} if center else {}
    try:
        expected = ref_op(inp_cpu, 8, hop_length=2, return_complex=True, **kwargs)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            (flag_gems.stft_center if center else flag_gems.stft)(
                inp_cpu.to(flag_gems.device),
                8,
                hop_length=2,
                return_complex=True,
                **kwargs,
            )
    else:
        actual = (flag_gems.stft_center if center else flag_gems.stft)(
            inp_cpu.to(flag_gems.device), 8, hop_length=2, return_complex=True, **kwargs
        )
        _assert_result(actual, expected, 8)


@pytest.mark.stft
@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize(
    "bad_input,n_fft,win_length",
    [
        ("rank3", 8, 8),
        ("bad_hop", 8, 8),
        ("bad_fft", 0, 0),
        ("bad_window", 8, 7),
    ],
)
def test_stft_invalid_inputs(center, bad_input, n_fft, win_length):
    shape = (2, 3, 32) if bad_input == "rank3" else (32,)
    inp_cpu = torch.randn(shape)
    window_cpu = torch.ones(6 if bad_input == "bad_window" else max(win_length, 1))
    kwargs = {"center": True, "pad_mode": "constant"} if center else {}
    if bad_input == "bad_hop":
        kwargs["hop_length"] = -1
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    with pytest.raises((RuntimeError, ValueError, TypeError)) as ref_error:
        ref_op(
            inp_cpu,
            n_fft,
            win_length=win_length,
            window=window_cpu,
            return_complex=True,
            **kwargs,
        )
    with pytest.raises(type(ref_error.value)):
        (flag_gems.stft_center if center else flag_gems.stft)(
            inp_cpu.to(flag_gems.device),
            n_fft,
            win_length=win_length,
            window=window_cpu.to(flag_gems.device),
            return_complex=True,
            **kwargs,
        )


@pytest.mark.stft_center
@pytest.mark.parametrize("align", [False, True])
def test_stft_center_rejects_explicit_align(align):
    inp_cpu = torch.randn(32)
    kwargs = dict(
        center=True,
        pad_mode="constant",
        align_to_window=align,
        return_complex=True,
    )
    with pytest.raises(RuntimeError, match="align_to_window"):
        torch.ops.aten.stft.center(inp_cpu, 8, **kwargs)
    with pytest.raises(RuntimeError, match="align_to_window"):
        flag_gems.stft_center(inp_cpu.to(flag_gems.device), 8, **kwargs)


@pytest.mark.stft
def test_stft_default_align_odd_window_native_boundary():
    # This ATen combination has had an as_strided bounds error. Preserve the
    # native result (or native error) rather than masking it as a passing value.
    inp_cpu = torch.randn(32)
    kwargs = dict(
        hop_length=4,
        win_length=7,
        window=torch.ones(7),
        align_to_window=True,
        return_complex=True,
    )
    try:
        expected = torch.ops.aten.stft.default(inp_cpu, 16, **kwargs)
    except RuntimeError:
        with pytest.raises(RuntimeError):
            flag_gems.stft(
                inp_cpu.to(flag_gems.device),
                16,
                **{**kwargs, "window": kwargs["window"].to(flag_gems.device)},
            )
    else:
        actual = flag_gems.stft(
            inp_cpu.to(flag_gems.device),
            16,
            **{**kwargs, "window": kwargs["window"].to(flag_gems.device)},
        )
        _assert_result(actual, expected, 16)


def _check_stft_fused_complex_window_gradients(dtype, center):
    n_fft = 256
    inp_cpu = _make_input((2, 775), dtype, noncontiguous=True)
    window_cpu = _make_window("complex", 128, torch.float32, noncontiguous=True)
    inp = _device_copy(inp_cpu)
    window = _device_copy(window_cpu).conj()
    if dtype.is_complex:
        inp = inp.conj()
    else:
        inp = torch._neg_view(inp)
    inp.requires_grad_()
    window.requires_grad_()
    ref_inp = inp.detach().cpu().to(_reference_dtype(dtype)).requires_grad_()
    ref_window = window.detach().cpu().to(torch.complex128).requires_grad_()
    kwargs = dict(
        hop_length=64,
        win_length=128,
        normalized=True,
        onesided=False,
        return_complex=True,
        align_to_window=None if center else True,
    )
    ref_op = torch.ops.aten.stft.center if center else torch.ops.aten.stft.default
    op = flag_gems.stft_center if center else flag_gems.stft
    if center:
        kwargs.update(center=True, pad_mode="reflect")
    expected = ref_op(ref_inp, n_fft, window=ref_window, **kwargs)
    actual = op(inp, n_fft, window=window, **kwargs)
    _assert_result(actual, expected.to(torch.complex64), n_fft)
    cotangent_cpu = torch.randn(actual.shape, dtype=torch.complex64)
    cotangent = torch._neg_view(cotangent_cpu.to(actual.device))
    refs = torch.autograd.grad(
        expected, (ref_inp, ref_window), -cotangent_cpu.to(expected.dtype)
    )
    grads = torch.autograd.grad(actual, (inp, window), cotangent)
    for grad, ref in zip(grads, refs):
        utils.gems_assert_close(grad.cpu(), ref, grad.dtype, reduce_dim=n_fft)


@pytest.mark.stft
@pytest.mark.parametrize("dtype", COMMON_DTYPES)
def test_stft_fused_complex_window_gradients(dtype):
    _check_stft_fused_complex_window_gradients(dtype, center=False)


@pytest.mark.stft_center
@pytest.mark.parametrize("dtype", COMMON_DTYPES)
def test_stft_center_fused_complex_window_gradients(dtype):
    _check_stft_fused_complex_window_gradients(dtype, center=True)


def _check_stft_empty_batch_gradients(n_fft, center):
    length = 2 * n_fft + 1
    hop = n_fft // 2
    inp = torch.empty((0, length), device=flag_gems.device, requires_grad=True)
    window = torch.ones(n_fft, device=flag_gems.device, requires_grad=True)
    kwargs = dict(hop_length=hop, window=window, return_complex=True)
    if center:
        kwargs.update(center=True, pad_mode="constant")
    op = flag_gems.stft_center if center else flag_gems.stft
    actual = op(inp, n_fft, **kwargs)
    frames = 1 + (length + (n_fft if center else 0) - n_fft) // hop
    assert actual.shape == (0, n_fft // 2 + 1, frames)
    assert actual.dtype == torch.complex64
    assert actual.device == inp.device
    dx, dw = torch.autograd.grad(actual, (inp, window), torch.empty_like(actual))
    assert dx.shape == inp.shape and dx.numel() == 0
    torch.testing.assert_close(dw.cpu(), torch.zeros(n_fft), atol=0, rtol=0)


@pytest.mark.stft
@pytest.mark.parametrize("n_fft", [8, 256])
def test_stft_empty_batch_gradients(n_fft):
    _check_stft_empty_batch_gradients(n_fft, center=False)


@pytest.mark.stft_center
@pytest.mark.parametrize("n_fft", [8, 256])
def test_stft_center_empty_batch_gradients(n_fft):
    _check_stft_empty_batch_gradients(n_fft, center=True)
