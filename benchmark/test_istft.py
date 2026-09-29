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

from . import base
from .generated_operator_utils import OperatorBenchmark

_COMPLEX_REAL_DTYPES = {
    torch.complex32: torch.float16,
    torch.complex64: torch.float32,
    torch.complex128: torch.float64,
}

# Static backend metadata, no runtime probe: complex128 needs fp64, and the
# complex32 lane runs through the cuFFT half-precision path, which is backend
# specific (the bfloat16 construction flag is not a half-precision FFT
# capability and is not used here).
_CUFFT_HALF = flag_gems.vendor_name == "nvidia"
ISTFT_DTYPES = [torch.complex64]
if _CUFFT_HALF:
    ISTFT_DTYPES.append(torch.complex32)
if flag_gems.runtime.device.support_fp64:
    ISTFT_DTYPES.append(torch.complex128)

# Performance spectrograms: freq_bins x n_frames (rank 2) and
# batch x freq_bins x n_frames (rank 3), all with freq_bins == n_fft // 2 + 1.
ISTFT_SHAPES = (
    (257, 256),
    (257, 1024),
    (257, 4096),
    (20, 257, 15),
    (16, 257, 256),
    (2, 513, 512),
    (8, 1025, 128),
)

# Shapes that additionally get explicit window descriptor rows, so the plan
# metadata can describe a provided analysis window next to the default rows where
# no window is built. hop_length=128 is the native default hop (n_fft // 4,
# independent of win_length); the ones row keeps its original explicit
# hop_length=256 == win_length // 2 configuration.
_WINDOW_DESCRIPTOR_SHAPES = ((257, 1024),)

# Explicit descriptor entries run by default: the shortest transform (one
# frequency bin, hop == win == 1, no edge trim), an even non-power-of-two n_fft
# with an explicit hop, and the same tiny transforms again with a complex output,
# whose transform kind the native op infers from the full-size frequency extent.
# This is the same descriptor path a custom shape file takes.
_DEFAULT_DESCRIPTORS = (
    {
        "spectrogram": [1, 20],
        "n_fft": 1,
        "hop_length": 1,
        "win_length": 1,
        "window": "ones",
        "center": False,
    },
    {"spectrogram": [51, 256], "n_fft": 100, "hop_length": 50},
    # At n_fft 1 and 2 the full-size extent and n_fft // 2 + 1 coincide, so the
    # native inference must be read as "two-sided unless the extent is the
    # one-sided one": both of these are measured positives on this backend.
    {
        "spectrogram": [1, 20],
        "n_fft": 1,
        "hop_length": 1,
        "win_length": 1,
        "window": "ones",
        "center": False,
        "return_complex": True,
    },
    {
        "spectrogram": [2, 256],
        "n_fft": 2,
        "hop_length": 1,
        "win_length": 2,
        "window": "ones",
        "center": False,
        "return_complex": True,
    },
)
DEFAULT_SHAPES = ISTFT_SHAPES + _DEFAULT_DESCRIPTORS

_WINDOW_KINDS = ("none", "hann", "ones")
_DESCRIPTOR_KEYS = (
    "n_fft",
    "hop_length",
    "win_length",
    "window",
    "center",
    "onesided",
    "normalized",
    "return_complex",
)


def _plan(
    shape,
    *,
    n_fft=None,
    hop_length=None,
    win_length=None,
    window="none",
    center=True,
    onesided=None,
    normalized=False,
    return_complex=False,
):
    """Describe one istft workload; every check here is metadata-only."""
    dims = tuple(shape)
    if len(dims) not in (2, 3):
        raise ValueError(
            f"istft expects a rank-2 or rank-3 spectrogram shape, got {dims}"
        )
    if any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in dims
    ):
        raise ValueError(
            f"istft spectrogram extents must be non-negative integers, got {dims}"
        )
    freq_bins, n_frames = dims[-2], dims[-1]
    # One frequency bin is the shortest valid transform (n_fft == 1) and only
    # exists with an explicit n_fft, which is validated below.
    if freq_bins < 1 or n_frames < 1:
        raise ValueError(
            f"istft needs at least 1 frequency bin and 1 frame, got {dims}"
        )
    if len(dims) == 3 and dims[0] < 1:
        raise ValueError(f"istft needs a non-empty batch extent, got {dims}")
    if n_fft is not None and (isinstance(n_fft, bool) or not isinstance(n_fft, int)):
        raise ValueError(f"istft n_fft must be an integer, got {n_fft!r}")
    if n_fft is None:
        n_fft = 2 * (freq_bins - 1)
    if n_fft <= 0:
        raise ValueError(f"istft needs a positive n_fft, got {n_fft}")
    if onesided is not None and not isinstance(onesided, bool):
        raise ValueError(f"istft onesided must be a bool or None, got {onesided!r}")
    for flag_name, flag_value in (
        ("center", center),
        ("normalized", normalized),
        ("return_complex", return_complex),
    ):
        # Strict bool: 0/1 or 0.0/1.0 would silently select a different call.
        if not isinstance(flag_value, bool):
            raise ValueError(f"istft {flag_name} must be a bool, got {flag_value!r}")
    if window not in _WINDOW_KINDS:
        raise ValueError(
            f"istft test window must be one of {_WINDOW_KINDS}, got {window!r}"
        )
    # The native op infers the transform kind from the frequency extent whenever
    # onesided is None, so both inferable forms are accepted and None stays None
    # in the call that is finally built.
    if onesided is None:
        if freq_bins not in (n_fft // 2 + 1, n_fft):
            raise ValueError(
                f"istft cannot infer onesided from freq_bins={freq_bins}, n_fft={n_fft}"
            )
    else:
        expected_bins = n_fft // 2 + 1 if onesided else n_fft
        if freq_bins != expected_bins:
            raise ValueError(
                f"istft needs freq_bins == {expected_bins} for n_fft={n_fft}, "
                f"got {freq_bins}"
            )
    # The effective transform kind, with the inference the native op applies when
    # onesided is None: a full-size frequency extent (freq_bins == n_fft) selects
    # the two-sided transform and any other accepted extent the one-sided one.
    # Measuring freq_bins == n_fft // 2 + 1 instead would misread the smallest
    # transforms, where n_fft 1 and 2 make both extents the same value: n_fft 1/2
    # with return_complex=True was measured to succeed on this backend, exactly
    # like an explicit onesided=False. A complex output needs the two-sided form.
    effective_onesided = freq_bins != n_fft if onesided is None else onesided
    if return_complex and effective_onesided:
        raise ValueError(
            "istft needs a two-sided transform for a complex output: "
            f"onesided={onesided!r}, freq_bins={freq_bins}, n_fft={n_fft}"
        )
    if win_length is not None and (
        isinstance(win_length, bool) or not isinstance(win_length, int)
    ):
        raise ValueError(f"istft win_length must be an integer, got {win_length!r}")
    if win_length is None:
        win_length = n_fft
    if hop_length is not None and (
        isinstance(hop_length, bool) or not isinstance(hop_length, int)
    ):
        raise ValueError(f"istft hop_length must be an integer, got {hop_length!r}")
    # Native default: hop_length == n_fft // 4, independent of win_length.
    if hop_length is None:
        hop_length = n_fft // 4
    if not 0 < hop_length <= win_length <= n_fft:
        raise ValueError(
            f"istft needs 0 < hop_length <= win_length <= n_fft, "
            f"got {hop_length}, {win_length}, {n_fft}"
        )
    call_kwargs = {
        "hop_length": hop_length,
        "win_length": win_length,
        # None means the built call passes no window; the kind string is only
        # recorded when a window tensor is actually constructed.
        "window": None if window == "none" else window,
        "center": center,
        "onesided": onesided,
        "normalized": normalized,
        "return_complex": return_complex,
    }
    return base.BenchmarkCasePlan(
        shape={"spectrogram": list(dims)},
        params={"spectrogram": list(dims), "n_fft": n_fft, **call_kwargs},
        builder_args=(dims, n_fft, call_kwargs),
    )


def _supported_for_dtype(plan, dtype):
    """Reject only the combinations the active backend cannot run for `dtype`."""
    if dtype is not torch.complex32:
        return True
    if not _CUFFT_HALF:
        return False
    _, n_fft, call_kwargs = plan.builder_args
    # Measured on this backend: cuFFT half precision is power-of-two only, and
    # the half lane has no complex-output path.
    if n_fft & (n_fft - 1):
        return False
    return not call_kwargs["return_complex"]


def _case_fn(shape, dtype):
    if isinstance(shape, dict):
        # Optional explicit descriptor entry, in a custom shape file or in the
        # defaults, e.g. {spectrogram: [51, 256], n_fft: 100, hop_length: 50} for
        # an even non-power-of-two n_fft.
        descriptor = dict(shape)
        dims = descriptor.pop("spectrogram")
        unknown = sorted(set(descriptor) - set(_DESCRIPTOR_KEYS))
        if unknown:
            raise ValueError(
                f"istft descriptor keys must be a subset of {_DESCRIPTOR_KEYS}, "
                f"got {unknown}"
            )
        plans = [_plan(dims, **descriptor)]
    else:
        plans = [_plan(shape)]
        if tuple(shape) in _WINDOW_DESCRIPTOR_SHAPES:
            plans.append(_plan(shape, hop_length=128, win_length=512, window="hann"))
            plans.append(_plan(shape, hop_length=256, win_length=512, window="ones"))
    for plan in plans:
        if _supported_for_dtype(plan, dtype):
            yield plan


def _build_inputs_fn(plan, dtype, device):
    dims, n_fft, call_kwargs = plan.builder_args
    real_dtype = _COMPLEX_REAL_DTYPES[dtype]
    # The shared complex branch only covers complex64, so the spectrogram is
    # built directly from its real component; a built window uses the matching
    # real dtype of that complex dtype.
    spectrogram = torch.complex(
        torch.randn(dims, dtype=real_dtype, device=device),
        torch.randn(dims, dtype=real_dtype, device=device),
    )
    window = None
    if call_kwargs["window"] == "hann":
        window = torch.hann_window(
            call_kwargs["win_length"], dtype=real_dtype, device=device
        )
    elif call_kwargs["window"] == "ones":
        window = torch.ones(call_kwargs["win_length"], dtype=real_dtype, device=device)
    return spectrogram, {"n_fft": n_fft, **call_kwargs, "window": window}


class IstftBenchmark(OperatorBenchmark):
    DEFAULT_SHAPES = DEFAULT_SHAPES
    DEFAULT_SHAPE_DESC = "freq_bins, n_frames"

    def set_shapes(self, shape_file_path=None):
        # Reuse the framework resolution (custom file: op name, then class name,
        # otherwise these defaults) instead of a private shape-file reader.
        OperatorBenchmark.set_shapes(
            self, shape_file_path, default_shapes=self.DEFAULT_SHAPES
        )


@pytest.mark.istft
def test_istft_benchmark():
    bench = IstftBenchmark(
        op_name="istft",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.istft,
        gems_op=getattr(flag_gems, "istft", None),
        dtypes=ISTFT_DTYPES,
    )
    bench.run()
