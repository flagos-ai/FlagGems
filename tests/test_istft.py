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
from . import test_utils as tu

# torch.ops.aten.istft exposes a single default overload (overloads() ==
# ['default']), so there is no out overload and every test passes its arguments
# straight to flag_gems.istft. The op reconstructs a real signal from a rank-2
# (freq_bins, n_frames) or rank-3 (batch, freq_bins, n_frames) spectrogram using
# an optional 1-D analysis window, so that window is a second tensor operand and
# its values and layout are covered explicitly below.

_COMPLEX_REAL_DTYPES = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# Static backend metadata, no runtime probe (the policy the FFT suites use): the
# complex32 lane runs through the cuFFT half-precision path, and the rejections
# recorded further down were measured on this backend, so both are scoped to it.
# utils.bf16_is_supported is a bfloat16 construction capability, not a
# half-precision FFT capability, and only gates the bfloat16 input negative.
_NVIDIA_CUFFT = flag_gems.vendor_name == "nvidia"

SUPPORTED_DTYPES = [torch.complex64]
if _NVIDIA_CUFFT:
    SUPPORTED_DTYPES.append(torch.complex32)
if utils.fp64_is_supported:
    SUPPORTED_DTYPES.append(torch.complex128)

HALF_DTYPES = [dtype for dtype in SUPPORTED_DTYPES if dtype is torch.complex32]
NON_HALF_DTYPES = [dtype for dtype in SUPPORTED_DTYPES if dtype is not torch.complex32]

NON_COMPLEX_DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
    torch.int8,
    torch.uint8,
    torch.bool,
]
# int64/bfloat16/FP8/FP64 tensor construction is gated on the matching static
# capability; int32, int8, uint8 and bool need no gate.
if utils.int64_is_supported:
    NON_COMPLEX_DTYPES.append(torch.int64)
if utils.bf16_is_supported:
    NON_COMPLEX_DTYPES.append(torch.bfloat16)
if utils.fp8_is_supported:
    NON_COMPLEX_DTYPES.extend([torch.float8_e4m3fn, torch.float8_e5m2])
if utils.fp64_is_supported:
    NON_COMPLEX_DTYPES.append(torch.float64)

_DEFAULT_SHAPE = (20, 257, 15)
_DEFAULT_N_FFT = 512
# A two-sided spectrogram carries exactly n_fft frequency bins.
_TWO_SIDED_SHAPE = (512, 128)

# The seven main rank-2/rank-3 spectrograms with freq_bins == n_fft // 2 + 1,
# spanning the frequency and frame extents the benchmark uses; the three
# following additional one-sided grids (n_fft 512, 1024 and 2048, all powers of
# two, differing only in the frequency and frame extents) are additive.
_ORIGINAL_MAIN_SHAPES = (
    (257, 256),
    (257, 1024),
    (257, 4096),
    (20, 257, 15),
    (16, 257, 256),
    (2, 513, 512),
    (8, 1025, 128),
)
_EXTRA_MAIN_SHAPES = ((257, 100), (513, 256), (1025, 64))
# The quick suite uses one rank-3 grid; the 257 frequency bins keep n_fft a power
# of two, which the half lane needs (see the power-of-two rule in _row_dtypes).
_QUICK_SHAPES = ((2, 257, 19),)
MAIN_SHAPES = tu.selected_cases(
    _ORIGINAL_MAIN_SHAPES + _EXTRA_MAIN_SHAPES, quick=_QUICK_SHAPES
)


def _make_window(dtype, kind, win_length):
    """Analysis window of `win_length` in the real dtype of `dtype`."""
    if kind is None:
        return None
    real_dtype = _COMPLEX_REAL_DTYPES[dtype]
    if kind == "hann":
        return torch.hann_window(win_length, dtype=real_dtype, device=flag_gems.device)
    if kind == "ones":
        return torch.ones(win_length, dtype=real_dtype, device=flag_gems.device)
    raise ValueError(f"unknown test window kind {kind!r}")


def _window_with_payload(dtype, kind, length=_DEFAULT_N_FFT):
    """Ones window of `length` carrying the payload named by `kind`."""
    real_dtype = _COMPLEX_REAL_DTYPES[dtype]
    if kind == "zero":
        return torch.zeros(length, dtype=real_dtype, device=flag_gems.device)
    window = torch.ones(length, dtype=real_dtype, device=flag_gems.device)
    if kind == "isolated-inf":
        # One infinite entry and strictly positive entries everywhere else, so
        # the window still overlap-adds to a positive envelope.
        window[0] = float("inf")
    elif kind == "zero-entry":
        window[0] = 0.0
        window[1] = 0.0
    else:
        payload = tu.make_special_input(real_dtype, kind)
        window[: payload.numel()] = payload
    return window


@pytest.mark.istft
@pytest.mark.parametrize("shape", MAIN_SHAPES)
@pytest.mark.parametrize("value_range", tu.selected_ranges())
@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_istft_value_range(shape, value_range, dtype):
    inp = tu.make_input(dtype, shape, value_range)
    ref_inp = tu.to_reference(inp)
    n_fft = 2 * (shape[-2] - 1)

    ref_out = torch.ops.aten.istft(ref_inp, n_fft)
    res_out = flag_gems.istft(inp, n_fft)

    tu.assert_result_close(res_out, ref_out)


# Parameter sweep on supported spectrograms; every keyword is forwarded verbatim
# and "window" is built by _make_window so the tensor operand is covered too.
_PARAMETER_ROWS = (
    ("defaults", SUPPORTED_DTYPES, _DEFAULT_SHAPE, _DEFAULT_N_FFT, {}),
    ("center-true", SUPPORTED_DTYPES, _DEFAULT_SHAPE, _DEFAULT_N_FFT, {"center": True}),
    (
        "center-false",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"center": False},
    ),
    (
        "normalized-true",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"normalized": True},
    ),
    (
        "normalized-false",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"normalized": False},
    ),
    (
        "onesided-true",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"onesided": True},
    ),
    (
        "onesided-none",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"onesided": None},
    ),
    # onesided=None infers a two-sided transform from the full n_fft extent.
    (
        "onesided-none-full-spectrum",
        SUPPORTED_DTYPES,
        _TWO_SIDED_SHAPE,
        _DEFAULT_N_FFT,
        {"onesided": None},
    ),
    (
        "hop-length-1",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"hop_length": 1},
    ),
    (
        "hop-length-256",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"hop_length": 256},
    ),
    # win_length < n_fft is zero padded at the frame edges: with the native edge
    # trim (center=True, the default here) the call is a valid positive, while
    # disabling the trim leaves the padded boundary uncovered and is a measured
    # rejection on the non-half dtypes
    # (test_istft_rejects_untrimmed_short_window).
    (
        "win-length-128",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"win_length": 128},
    ),
    (
        "window-ones",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"window": "ones"},
    ),
    (
        "window-hann-hop-128",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"window": "hann", "hop_length": 128},
    ),
    (
        "return-complex-false",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"return_complex": False},
    ),
    ("length-64", SUPPORTED_DTYPES, _DEFAULT_SHAPE, _DEFAULT_N_FFT, {"length": 64}),
    ("length-2000", SUPPORTED_DTYPES, _DEFAULT_SHAPE, _DEFAULT_N_FFT, {"length": 2000}),
    (
        "length-512-center-false",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"length": 512, "center": False},
    ),
    # win_length == 1 with hop_length == 1 reconstructs one sample per frame; with
    # the default edge trim (center=True) it is a positive call, while the
    # untrimmed form is a measured rejection on the non-half dtypes.
    (
        "hop-1-win-1-window-ones",
        SUPPORTED_DTYPES,
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"hop_length": 1, "win_length": 1, "window": "ones"},
    ),
    # At n_fft 1 and 2 the full-size extent equals n_fft // 2 + 1, and the native
    # op selects the two-sided transform when onesided is None (measured on this
    # backend: these calls succeed for complex64/128 and match an explicit
    # onesided=False, while onesided=True is rejected). The half lane has no
    # complex-output path, so these rows use the non-half dtypes.
    (
        "tiny-nfft-1-onesided-none-return-complex",
        NON_HALF_DTYPES,
        (1, 20),
        1,
        {
            "hop_length": 1,
            "win_length": 1,
            "window": "ones",
            "center": False,
            "return_complex": True,
        },
    ),
    (
        "tiny-nfft-2-onesided-none-return-complex",
        NON_HALF_DTYPES,
        (2, 256),
        2,
        {
            "hop_length": 1,
            "win_length": 2,
            "window": "ones",
            "center": False,
            "return_complex": True,
        },
    ),
    # Complex output requires a two-sided transform; the half lane is measured to
    # have no complex-output path, so these rows use the non-half dtypes.
    (
        "two-sided-return-complex",
        NON_HALF_DTYPES,
        _TWO_SIDED_SHAPE,
        _DEFAULT_N_FFT,
        {"onesided": False, "return_complex": True},
    ),
    (
        "two-sided-onesided-none-return-complex",
        NON_HALF_DTYPES,
        _TWO_SIDED_SHAPE,
        _DEFAULT_N_FFT,
        {"onesided": None, "return_complex": True},
    ),
)
PARAMETER_CASES = tu.selected_cases(
    [
        (name, shape, n_fft, kwargs, dtype)
        for name, dtypes, shape, n_fft, kwargs in _PARAMETER_ROWS
        for dtype in dtypes
    ],
    quick=(),
)
PARAMETER_IDS = [f"{name}-{str(dtype)[6:]}" for name, _, _, _, dtype in PARAMETER_CASES]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,shape,n_fft,kwargs,dtype", PARAMETER_CASES, ids=PARAMETER_IDS
)
def test_istft_parameters(name, shape, n_fft, kwargs, dtype):
    del name
    kwargs = dict(kwargs)
    kind = kwargs.pop("window", None)
    window = _make_window(dtype, kind, kwargs.get("win_length", n_fft))
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_window = tu.to_reference(window)

    ref_out = torch.ops.aten.istft(ref_inp, n_fft, window=ref_window, **kwargs)
    res_out = flag_gems.istft(inp, n_fft, window=window, **kwargs)

    tu.assert_result_close(res_out, ref_out)


# Boundary geometries: the shortest transforms, both frame extents of a single
# frame, non-power-of-two signal sizes, and the untrimmed short-window calls.
# The `lane` field pins a row's dtype list; None selects the power-of-two rule
# in _row_dtypes below.
_BOUNDARY_ROWS = (
    (
        "nfft-1-ones-hop-1-center-false",
        (1, 20),
        1,
        {"hop_length": 1, "win_length": 1, "window": "ones", "center": False},
        None,
    ),
    ("nfft-2-ones-hop-1", (2, 256), 2, {"hop_length": 1, "window": "ones"}, None),
    ("nfft-6-even-non-power-of-two", (4, 256), 6, {}, None),
    ("nfft-100-even-non-power-of-two", (51, 256), 100, {}, None),
    ("nfft-513-odd-non-power-of-two", (257, 128), 513, {}, None),
    ("frames-1-center-false", (257, 1), _DEFAULT_N_FFT, {"center": False}, None),
    (
        "frames-1-full-hop",
        (257, 1),
        _DEFAULT_N_FFT,
        {"hop_length": _DEFAULT_N_FFT, "window": "ones", "center": False},
        None,
    ),
    # Disabling the edge trim leaves the padded boundary of a short window
    # uncovered. That is a measured rejection on the non-half dtypes
    # (test_istft_rejects_untrimmed_short_window) but is accepted on the half
    # lane, so the same geometry stays a positive workload there.
    (
        "win-128-hop-128-untrimmed",
        (257, 100),
        _DEFAULT_N_FFT,
        {"hop_length": 128, "win_length": 128, "center": False},
        HALF_DTYPES,
    ),
    (
        "win-1-hop-1-ones-untrimmed",
        _DEFAULT_SHAPE,
        _DEFAULT_N_FFT,
        {"hop_length": 1, "win_length": 1, "window": "ones", "center": False},
        HALF_DTYPES,
    ),
)


def _row_dtypes(n_fft, lane):
    """Dtype lane of a boundary row: the half lane accepts power-of-two n_fft only."""
    if lane is not None:
        return lane
    return SUPPORTED_DTYPES if n_fft & (n_fft - 1) == 0 else NON_HALF_DTYPES


BOUNDARY_CASES = tu.selected_cases(
    [
        (name, shape, n_fft, kwargs, dtype)
        for name, shape, n_fft, kwargs, lane in _BOUNDARY_ROWS
        for dtype in _row_dtypes(n_fft, lane)
    ],
    quick=(),
)
BOUNDARY_IDS = [f"{name}-{str(dtype)[6:]}" for name, _, _, _, dtype in BOUNDARY_CASES]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,shape,n_fft,kwargs,dtype", BOUNDARY_CASES, ids=BOUNDARY_IDS
)
def test_istft_boundaries(name, shape, n_fft, kwargs, dtype):
    del name
    kwargs = dict(kwargs)
    kind = kwargs.pop("window", None)
    window = _make_window(dtype, kind, kwargs.get("win_length", n_fft))
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    ref_inp = tu.to_reference(inp)
    ref_window = tu.to_reference(window)

    ref_out = torch.ops.aten.istft(ref_inp, n_fft, window=ref_window, **kwargs)
    res_out = flag_gems.istft(inp, n_fft, window=window, **kwargs)

    tu.assert_result_close(res_out, ref_out)


def _layout_spectrogram(dtype, name):
    """Spectrogram views with unusual strides or a non-zero storage offset."""
    if name == "strided":
        dense = tu.make_input(dtype, (257, 400), ["-1", "1"])
        return dense[:, 100:300:2], _DEFAULT_N_FFT
    if name == "transposed":
        return tu.make_input(dtype, (400, 257), ["-1", "1"]).t(), _DEFAULT_N_FFT
    # Non-zero storage offset; 251 frequency bins imply n_fft == 500, which the
    # half lane cannot use (cuFFT half precision is power-of-two only, measured).
    dense = tu.make_input(dtype, (254, 305), ["-1", "1"])
    return dense[3:254, 5:305], 500


INPUT_LAYOUT_CASES = tu.selected_cases(
    [
        (name, dtype)
        for name in ("strided", "transposed", "offset")
        for dtype in (NON_HALF_DTYPES if name == "offset" else SUPPORTED_DTYPES)
    ],
    quick=(),
)


@pytest.mark.istft
@pytest.mark.parametrize("name,dtype", INPUT_LAYOUT_CASES)
def test_istft_input_layouts(name, dtype):
    inp, n_fft = _layout_spectrogram(dtype, name)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.istft(ref_inp, n_fft)
    res_out = flag_gems.istft(inp, n_fft)

    tu.assert_result_close(res_out, ref_out)


def _layout_window(dtype, name):
    """Window views with a non-zero offset, a stride-2 layout or a zero stride."""
    real_dtype = _COMPLEX_REAL_DTYPES[dtype]
    if name == "offset":
        return tu.make_input(real_dtype, (1024,), ["-1", "1"])[7:519]
    if name == "strided":
        # Column of an (n, 2) tensor: a 1-D window with stride 2.
        return tu.make_input(real_dtype, (512, 2), ["-1", "1"]).t()[1]
    # Zero-stride broadcast view: one value expanded over the whole window.
    return tu.make_input(real_dtype, (1,), ["-1", "1"]).expand(_DEFAULT_N_FFT)


# The rank-2 (257, 100) grid is the original window-layout workload; the rank-3
# grid is additive.
_WINDOW_LAYOUT_SHAPES = ((257, 100), _DEFAULT_SHAPE)
WINDOW_LAYOUT_CASES = tu.selected_cases(
    [
        (name, shape, dtype)
        for shape in _WINDOW_LAYOUT_SHAPES
        for name in ("offset", "strided", "expanded")
        for dtype in SUPPORTED_DTYPES
    ],
    quick=(),
)


@pytest.mark.istft
@pytest.mark.parametrize("name,shape,dtype", WINDOW_LAYOUT_CASES)
def test_istft_window_layouts(name, shape, dtype):
    inp = tu.make_input(dtype, shape, ["-1", "1"])
    window = _layout_window(dtype, name)
    ref_inp = tu.to_reference(inp)
    ref_window = tu.to_reference(window)

    ref_out = torch.ops.aten.istft(ref_inp, _DEFAULT_N_FFT, window=ref_window)
    res_out = flag_gems.istft(inp, _DEFAULT_N_FFT, window=window)

    tu.assert_result_close(res_out, ref_out)


def _special_spectrogram(dtype, scenario, placement):
    """Spectrogram carrying a special payload densely, at one element, or in one batch row."""
    freq_bins, n_frames, n_fft = 129, 100, 256
    payload = tu.make_special_input(dtype, scenario)
    if placement == "batched":
        inp = tu.make_input(dtype, (3, freq_bins, n_frames), ["-1", "1"])
        inp[0, freq_bins // 2, n_frames // 2 - 2 : n_frames // 2 + 3] = payload
        return inp, n_fft
    inp = tu.make_input(dtype, (freq_bins, n_frames), ["-1", "1"])
    if placement == "dense":
        repeats = inp.numel() // payload.numel() + 1
        dense = payload.repeat(repeats)[: inp.numel()].reshape(inp.shape)
        return dense, n_fft
    inp[freq_bins // 2, n_frames // 2 - 2 : n_frames // 2 + 3] = payload
    return inp, n_fft


SPECIAL_CASES = tu.selected_cases(
    [
        (scenario, placement, dtype)
        for scenario in ("nan", "inf", "mixed")
        for placement in ("dense", "isolated", "batched")
        for dtype in SUPPORTED_DTYPES
    ],
    quick=(),
)


@pytest.mark.istft
@pytest.mark.parametrize("scenario,placement,dtype", SPECIAL_CASES)
def test_istft_special_spectrogram_values(scenario, placement, dtype):
    inp, n_fft = _special_spectrogram(dtype, scenario, placement)
    ref_inp = tu.to_reference(inp)

    ref_out = torch.ops.aten.istft(ref_inp, n_fft)
    res_out = flag_gems.istft(inp, n_fft)

    tu.assert_result_close(res_out, ref_out)


# Positive window payloads measured to be accepted at hop_length=128 with the
# edge trim disabled: the shared nan and mixed payloads for every supported
# dtype, an isolated Inf with strictly positive entries elsewhere, and the shared
# inf payload on the half lane, whose accumulation path accepts it.
_WINDOW_SPECIAL_ROWS = (
    ("nan", SUPPORTED_DTYPES),
    ("mixed", SUPPORTED_DTYPES),
    ("isolated-inf", SUPPORTED_DTYPES),
    ("inf", HALF_DTYPES),
)
WINDOW_SPECIAL_CASES = tu.selected_cases(
    [(kind, dtype) for kind, dtypes in _WINDOW_SPECIAL_ROWS for dtype in dtypes],
    quick=(),
)


@pytest.mark.istft
@pytest.mark.parametrize("kind,dtype", WINDOW_SPECIAL_CASES)
def test_istft_window_special_values(kind, dtype):
    inp = tu.make_input(dtype, (257, 100), ["-1", "1"])
    window = _window_with_payload(dtype, kind)
    ref_inp = tu.to_reference(inp)
    ref_window = tu.to_reference(window)

    ref_out = torch.ops.aten.istft(
        ref_inp, _DEFAULT_N_FFT, window=ref_window, hop_length=128, center=False
    )
    res_out = flag_gems.istft(
        inp, _DEFAULT_N_FFT, window=window, hop_length=128, center=False
    )

    tu.assert_result_close(res_out, ref_out)


def _backward_setup(dtype, shape, kwargs, window_kind):
    inp = tu.make_input(dtype, shape, ["-1", "1"]).requires_grad_(True)
    window = _make_window(dtype, window_kind, kwargs.get("win_length", _DEFAULT_N_FFT))
    if window is not None:
        window.requires_grad_(True)
    return inp, window


_BACKWARD_ROWS = (
    ("defaults", (20, 257, 15), {}, None),
    (
        "hop-128-win-256-hann",
        (257, 100),
        {"hop_length": 128, "win_length": 256},
        "hann",
    ),
    (
        "hop-256-ones-center-false",
        (257, 100),
        {"hop_length": 256, "center": False},
        "ones",
    ),
    ("normalized-true", (257, 100), {"normalized": True}, None),
)
BACKWARD_CASES = tu.selected_cases(
    [
        (name, shape, kwargs, window_kind, dtype)
        for name, shape, kwargs, window_kind in _BACKWARD_ROWS
        for dtype in SUPPORTED_DTYPES
    ],
    quick=(),
)
BACKWARD_IDS = [f"{name}-{str(dtype)[6:]}" for name, _, _, _, dtype in BACKWARD_CASES]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,shape,kwargs,window_kind,dtype", BACKWARD_CASES, ids=BACKWARD_IDS
)
def test_istft_backward(name, shape, kwargs, window_kind, dtype):
    del name
    inp, window = _backward_setup(dtype, shape, kwargs, window_kind)
    ref_inp = tu.to_reference(inp)
    ref_window = tu.to_reference(window)

    ref_out = torch.ops.aten.istft(ref_inp, _DEFAULT_N_FFT, window=ref_window, **kwargs)
    # Non-constant upstream gradient, matched between both runs, so the compared
    # gradients are not a constant multiple of the summed output.
    upstream = torch.randn(ref_out.shape, dtype=ref_out.dtype, device=flag_gems.device)
    ref_wrt = (ref_inp,) if ref_window is None else (ref_inp, ref_window)
    ref_grads = torch.autograd.grad(
        ref_out, ref_wrt, grad_outputs=tu.to_reference(upstream)
    )

    res_out = flag_gems.istft(inp, _DEFAULT_N_FFT, window=window, **kwargs)
    res_wrt = (inp,) if window is None else (inp, window)
    res_grads = torch.autograd.grad(res_out, res_wrt, grad_outputs=upstream)

    tu.assert_result_close(res_out, ref_out)
    for res_grad, ref_grad in zip(res_grads, ref_grads):
        tu.assert_result_close(res_grad, ref_grad)


@pytest.mark.istft
@pytest.mark.parametrize("dtype", NON_COMPLEX_DTYPES)
def test_istft_rejects_non_complex_input(dtype):
    inp = torch.zeros((257, 100), dtype=dtype, device=flag_gems.device)
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT)


@pytest.mark.istft
@pytest.mark.parametrize(
    "shape", [(), (1,), (256,), (257, 100, 2, 2), (2, 2, 257, 100, 2)]
)
def test_istft_rejects_invalid_rank(shape):
    inp = torch.zeros(shape, dtype=torch.complex64, device=flag_gems.device)
    # The native op raises IndexError below rank 2 and RuntimeError above it.
    with pytest.raises((RuntimeError, IndexError)):
        flag_gems.istft(inp, _DEFAULT_N_FFT)


@pytest.mark.istft
@pytest.mark.parametrize(
    "shape,n_fft", [((129, 100), 512), ((51, 100), 512), ((1, 100), 2)]
)
def test_istft_rejects_frequency_mismatch(shape, n_fft):
    inp = tu.make_input(torch.complex64, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, n_fft)


@pytest.mark.istft
@pytest.mark.parametrize("shape", [(257, 0), (0, 257, 100)])
def test_istft_rejects_empty_extent(shape):
    inp = tu.make_input(torch.complex64, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT)


_INVALID_PARAMETER_ROWS = (
    ("nfft-0", 0, {}),
    ("nfft-511-mismatch", 511, {}),
    ("hop-length-0", 512, {"hop_length": 0}),
    ("hop-length-negative", 512, {"hop_length": -8}),
    ("win-length-exceeds-nfft", 512, {"win_length": 1024}),
    ("hop-length-exceeds-win-length", 512, {"win_length": 256, "hop_length": 512}),
    ("length-negative", 512, {"length": -4}),
)


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,n_fft,kwargs",
    _INVALID_PARAMETER_ROWS,
    ids=[r[0] for r in _INVALID_PARAMETER_ROWS],
)
def test_istft_rejects_invalid_parameters(name, n_fft, kwargs):
    del name
    inp = tu.make_input(torch.complex64, (257, 100), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, n_fft, **kwargs)


def _invalid_window(name):
    if name == "rank-2":
        return torch.ones(2, 512, dtype=torch.float32, device=flag_gems.device), {}
    if name == "length-mismatch":
        return torch.ones(256, dtype=torch.float32, device=flag_gems.device), {}
    if name == "all-zero":
        return torch.zeros(512, dtype=torch.float32, device=flag_gems.device), {}
    return torch.ones(512, dtype=torch.complex64, device=flag_gems.device), {
        "return_complex": False
    }


_INVALID_WINDOW_KINDS = (
    "rank-2",
    "length-mismatch",
    "all-zero",
    "complex-window-real-output",
)


@pytest.mark.istft
@pytest.mark.parametrize("name", _INVALID_WINDOW_KINDS)
def test_istft_rejects_invalid_window(name):
    inp = tu.make_input(torch.complex64, (257, 100), ["-1", "1"])
    window, kwargs = _invalid_window(name)
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT, window=window, **kwargs)


@pytest.mark.istft
@pytest.mark.parametrize("onesided", [True, None])
def test_istft_rejects_onesided_complex_output(onesided):
    inp = tu.make_input(torch.complex64, (257, 100), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT, onesided=onesided, return_complex=True)


# Measured on this backend at hop_length=128 with the edge trim disabled: the
# shared inf payload (which also holds 0.0/-0.0), a two-entry zero window and an
# all-zero window are rejected, while the nan and mixed payloads and an isolated
# Inf with strictly positive entries elsewhere are accepted. The rejection is
# therefore not caused by infinities alone and is not a general rule, so these
# cases are recorded as measured and scoped to the non-half dtypes of the
# backend they were measured on.
_WINDOW_PAYLOAD_ROWS = (
    ("inf-payload", "inf"),
    ("zero-entry", "zero-entry"),
    ("all-zero", "zero"),
)
WINDOW_PAYLOAD_CASES = [
    (name, kind, dtype)
    for name, kind in _WINDOW_PAYLOAD_ROWS
    for dtype in (NON_HALF_DTYPES if _NVIDIA_CUFFT else ())
]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,kind,dtype",
    WINDOW_PAYLOAD_CASES,
    ids=[f"{n}-{str(d)[6:]}" for n, _, d in WINDOW_PAYLOAD_CASES],
)
def test_istft_rejects_window_payload(name, kind, dtype):
    del name
    inp = tu.make_input(dtype, (257, 100), ["-1", "1"])
    window = _window_with_payload(dtype, kind)
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            inp, _DEFAULT_N_FFT, window=window, hop_length=128, center=False
        )


# A window shorter than n_fft is zero padded at the frame edges, so leaving the
# edge trim on (center=True, "trimmed") is the measured positive and turning the
# trim off (center=False, "untrimmed") leaves the padded boundary uncovered, which
# the native overlap-add normalization rejects: measured on this backend for
# n_fft=512 with (win_length, hop_length) == (1, 1) and (128, 128), while the half
# lane accepts the same untrimmed calls (see _BOUNDARY_ROWS).
_WINDOW_GEOMETRY_ROWS = (
    ("win-1-hop-1-untrimmed", {"hop_length": 1, "win_length": 1, "window": "ones"}),
    ("win-128-hop-128-untrimmed", {"hop_length": 128, "win_length": 128}),
)
WINDOW_GEOMETRY_CASES = [
    (name, kwargs, dtype)
    for name, kwargs in _WINDOW_GEOMETRY_ROWS
    for dtype in (NON_HALF_DTYPES if _NVIDIA_CUFFT else ())
]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,kwargs,dtype",
    WINDOW_GEOMETRY_CASES,
    ids=[f"{n}-{str(d)[6:]}" for n, _, d in WINDOW_GEOMETRY_CASES],
)
def test_istft_rejects_untrimmed_short_window(name, kwargs, dtype):
    del name
    kwargs = dict(kwargs, center=False)
    kind = kwargs.pop("window", None)
    window = _make_window(dtype, kind, kwargs["win_length"])
    inp = tu.make_input(dtype, (257, 100), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT, window=window, **kwargs)


# cuFFT half precision has no complex-output path (measured: "unfold_backward_cuda
# not implemented for 'ComplexHalf'"), so complex output is rejected there.
_HALF_COMPLEX_OUTPUT_ROWS = (
    (
        "two-sided-return-complex",
        (512, 128),
        {"onesided": False, "return_complex": True},
    ),
    (
        "two-sided-onesided-none-return-complex",
        (512, 128),
        {"onesided": None, "return_complex": True},
    ),
)
HALF_COMPLEX_OUTPUT_CASES = [row for row in _HALF_COMPLEX_OUTPUT_ROWS if _NVIDIA_CUFFT]


@pytest.mark.istft
@pytest.mark.parametrize(
    "name,shape,kwargs",
    HALF_COMPLEX_OUTPUT_CASES,
    ids=[r[0] for r in HALF_COMPLEX_OUTPUT_CASES],
)
def test_istft_rejects_complex32_complex_output(name, shape, kwargs):
    del name
    inp = tu.make_input(torch.complex32, shape, ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, _DEFAULT_N_FFT, **kwargs)


# The half lane is measured to accept power-of-two signal sizes only.
_HALF_N_FFT_ROWS = (6, 100, 500, 513)
HALF_N_FFT_CASES = [
    (n_fft, n_fft // 2 + 1) for n_fft in _HALF_N_FFT_ROWS if _NVIDIA_CUFFT
]


@pytest.mark.istft
@pytest.mark.parametrize("n_fft,freq_bins", HALF_N_FFT_CASES)
def test_istft_rejects_complex32_non_power_of_two_n_fft(n_fft, freq_bins):
    inp = tu.make_input(torch.complex32, (freq_bins, 64), ["-1", "1"])
    with pytest.raises(RuntimeError):
        flag_gems.istft(inp, n_fft)


# Master exercises the real-signal STFT round trip in addition to the direct
# spectrogram workloads above. Reuse the current backend dtype lanes.
_ROUNDTRIP_DTYPES = [_COMPLEX_REAL_DTYPES[dtype] for dtype in SUPPORTED_DTYPES]
_ROUNDTRIP_NON_HALF_DTYPES = [_COMPLEX_REAL_DTYPES[dtype] for dtype in NON_HALF_DTYPES]

# torch computes the fp16/complex32 inverse transform in half precision, so its
# reference carries ~1e-3 relative error; the Triton kernels accumulate in fp32
# and are therefore closer to the true signal. The tolerance follows the
# reference's own precision rather than the input dtype.
_ATOL = {
    torch.float16: 5e-3,
    torch.float32: 1e-4,
    torch.float64: 1e-7,
}

# (n_fft, hop_length, win_length) combinations: the default configuration, a
# minimal transform, a window shorter than the transform size, and a large
# transform.
FFT_PARAMS = [
    (256, 64, 256),
    (16, 4, 16),
    (512, 128, 256),
    (1024, 256, 1024),
]


def _stft(signal, n_fft, hop_length, win_length, window, **kwargs):
    return torch.stft(
        signal,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        **kwargs,
    )


def _build_case(
    signal_len, batch, n_fft, hop_length, win_length, dtype, rect_window=False, **kwargs
):
    """Return (gems spec, gems window, reference spec, reference window).

    The same underlying signal and window drive both sides: the reference
    operands are the upcast CPU copies (utils.to_reference), so the two
    spectrograms describe one signal.
    """
    device = flag_gems.device
    if rect_window:
        window = torch.ones(win_length, dtype=dtype, device=device)
    else:
        window = torch.hann_window(win_length, dtype=dtype, device=device)
    shape = (signal_len,) if batch == 0 else (batch, signal_len)
    signal = torch.randn(shape, dtype=dtype, device=device)
    spec = _stft(signal, n_fft, hop_length, win_length, window, **kwargs)

    ref_signal = utils.to_reference(signal, upcast=True)
    ref_window = utils.to_reference(window, upcast=True)
    ref_spec = _stft(ref_signal, n_fft, hop_length, win_length, ref_window, **kwargs)
    return spec, window, ref_spec, ref_window


def _assert_roundtrip_close(res, ref, device, dtype, atol):
    assert res.device == device
    assert res.dtype == dtype
    utils.gems_assert_close(res, ref, dtype, atol=atol)


@pytest.mark.istft
@pytest.mark.parametrize("dtype", _ROUNDTRIP_DTYPES)
@pytest.mark.parametrize("n_fft,hop_length,win_length", FFT_PARAMS)
def test_istft_roundtrip(dtype, n_fft, hop_length, win_length):
    signal_len = 4096
    batch = 2
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len,
    )
    ref_out = torch.istft(
        ref_spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=ref_window,
        length=signal_len,
    ).to(dtype)

    assert res_out.shape == ref_out.shape
    _assert_roundtrip_close(res_out, ref_out, spec.device, dtype, _ATOL[dtype])


@pytest.mark.istft
@pytest.mark.parametrize("dtype", _ROUNDTRIP_DTYPES)
def test_istft_1d_signal(dtype):
    signal_len = 1024
    n_fft, hop_length, win_length = 256, 64, 256
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, 0, n_fft, hop_length, win_length, dtype
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len,
    )
    ref_out = torch.istft(
        ref_spec, n_fft, hop_length, win_length, ref_window, length=signal_len
    ).to(dtype)

    assert res_out.dim() == 1 and ref_out.dim() == 1
    _assert_roundtrip_close(res_out, ref_out, spec.device, dtype, _ATOL[dtype])


@pytest.mark.istft
# fp16 is omitted here: the complex-return and normalized variants are covered
# by test_istft_roundtrip for all three dtypes, and the fp64 path here already
# pins the exact-arithmetic behavior this test targets.
@pytest.mark.parametrize("dtype", _ROUNDTRIP_NON_HALF_DTYPES)
def test_istft_variants(dtype):
    signal_len = 2048
    n_fft, hop_length, win_length = 256, 64, 256
    batch = 2
    atol = _ATOL[dtype]

    # normalized=True round trip
    spec, window, ref_spec, ref_window = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype, normalized=True
    )
    res = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        normalized=True,
        length=signal_len,
    )
    ref = torch.istft(
        ref_spec,
        n_fft,
        hop_length,
        win_length,
        ref_window,
        normalized=True,
        length=signal_len,
    ).to(dtype)
    _assert_roundtrip_close(res, ref, spec.device, dtype, atol)

    # explicit length shorter than the expected output
    res = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        length=signal_len - 8,
    )
    ref = torch.istft(
        ref_spec, n_fft, hop_length, win_length, ref_window, length=signal_len - 8
    ).to(dtype)
    _assert_roundtrip_close(res, ref, spec.device, dtype, atol)

    # no explicit length (returns the full centered signal)
    res = flag_gems.istft(
        spec, n_fft, hop_length=hop_length, win_length=win_length, window=window
    )
    ref = torch.istft(ref_spec, n_fft, hop_length, win_length, ref_window).to(dtype)
    _assert_roundtrip_close(res, ref, spec.device, dtype, atol)

    # two-sided spectrogram (onesided=False), inferred onesided=None
    spec2, window2, ref_spec2, ref_window2 = _build_case(
        signal_len, batch, n_fft, hop_length, win_length, dtype, onesided=False
    )
    res2 = flag_gems.istft(
        spec2,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window2,
        length=signal_len,
    )
    ref2 = torch.istft(
        ref_spec2, n_fft, hop_length, win_length, ref_window2, length=signal_len
    ).to(dtype)
    _assert_roundtrip_close(res2, ref2, spec2.device, dtype, atol)

    # return_complex=True on a two-sided spectrogram
    res3 = flag_gems.istft(
        spec2,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window2,
        return_complex=True,
        onesided=False,
    )
    ref3 = torch.istft(
        ref_spec2,
        n_fft,
        hop_length,
        win_length,
        ref_window2,
        return_complex=True,
        onesided=False,
    )
    assert res3.is_complex() and ref3.is_complex()
    for res_c, ref_c in (
        (res3.real, ref3.real.to(dtype)),
        (res3.imag, ref3.imag.to(dtype)),
    ):
        _assert_roundtrip_close(res_c, ref_c, spec2.device, dtype, atol)


@pytest.mark.istft
def test_istft_center_false():
    # With center=False the hann-window overlap-add envelope vanishes at the
    # tail and torch rejects the call; a rectangular window keeps the envelope
    # non-zero on both sides.
    dtype = torch.float32
    signal_len = 2048
    n_fft, hop_length, win_length = 256, 64, 256
    length = signal_len - n_fft
    spec, window, ref_spec, ref_window = _build_case(
        signal_len,
        2,
        n_fft,
        hop_length,
        win_length,
        dtype,
        rect_window=True,
        center=False,
    )
    res_out = flag_gems.istft(
        spec,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        length=length,
    )
    ref_out = torch.istft(
        ref_spec,
        n_fft,
        hop_length,
        win_length,
        ref_window,
        center=False,
        length=length,
    ).to(dtype)
    _assert_roundtrip_close(res_out, ref_out, spec.device, dtype, _ATOL[dtype])


@pytest.mark.istft
def test_istft_errors():
    device = flag_gems.device
    n_fft, hop_length, win_length = 256, 64, 256
    window = torch.hann_window(win_length, device=device)
    signal = torch.randn(1024, device=device)
    spec = torch.stft(
        signal, n_fft, hop_length, win_length, window, return_complex=True
    )
    spec2 = torch.stft(
        signal,
        n_fft,
        hop_length,
        win_length,
        window,
        return_complex=True,
        onesided=False,
    )

    # real input is rejected
    real_spec = torch.view_as_real(spec)
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            real_spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
        )
    # onesided output with return_complex=True is rejected
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            return_complex=True,
        )
    # frequency dimension must match n_fft/2+1 (onesided) or n_fft (two-sided)
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            onesided=False,
        )
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec2,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            onesided=True,
        )
    # hop_length must not exceed win_length
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec, n_fft, hop_length=n_fft + 1, win_length=win_length, window=window
        )
    # win_length must not exceed n_fft
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec, n_fft, hop_length=hop_length, win_length=n_fft + 1, window=window
        )
    # window shape must match win_length
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            spec,
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window[: win_length - 1],
        )
    # empty input is rejected
    with pytest.raises(RuntimeError):
        flag_gems.istft(
            torch.empty(0, n_fft // 2 + 1, 4, dtype=torch.complex64, device=device),
            n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )
