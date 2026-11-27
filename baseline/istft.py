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

"""NVIDIA native baseline for `istft`.

Mirrors benchmark/test_istft.py (IstftBenchmark, a base.Benchmark subclass):
    op_name  = "istft"
    torch_op = torch.istft
    dtypes   = [torch.float32]

The class overrides set_shapes with ISTFT_SHAPES and declares shape_desc
"batch, signal length, n_fft, hop_length", so core_shapes.yaml is not
consulted.

The spectrum comes from a real torch.stft of a random signal (hann window,
center=True, return_complex=True), which build_inputs performs outside the
timed region as the benchmark's get_input_iter does. All arguments are
positional: (spec, n_fft, hop, n_fft, window, True, False, None, None, False)
-- center=True, normalized=False, onesided=None, length=None,
return_complex=False.

float32 only: cuFFT accepts complex64/complex128 spectra, and the
fp16/complex32 path through torch.stft is experimental and much slower.
"""

import torch

OP_NAME = "istft"
DTYPES = [torch.float32]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = "torch.istft is unavailable on this torch build"

# test_istft.py::ISTFT_SHAPES -- (batch, signal length, n_fft, hop_length)
_SHAPES = [
    (1, 4096, 256, 64),
    (8, 4096, 256, 64),
    (32, 4096, 256, 64),
    (8, 16384, 256, 64),
    (8, 16384, 1024, 256),
    (1, 65536, 512, 128),
]

_CENTER = True
_NORMALIZED = False


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def native():
    op = _resolve("torch.istft")
    return op if callable(op) else None


def grid():
    return [
        {"batch": b, "signal_len": s, "n_fft": n, "hop": h} for b, s, n, h in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    batch = binding["batch"]
    signal_len = binding["signal_len"]
    n_fft = binding["n_fft"]
    hop = binding["hop"]
    window = torch.hann_window(n_fft, dtype=dtype, device=device)
    signal = torch.randn(batch, signal_len, dtype=dtype, device=device)
    spec = torch.stft(
        signal,
        n_fft,
        hop_length=hop,
        win_length=n_fft,
        window=window,
        center=_CENTER,
        return_complex=True,
    )
    return (
        spec,
        n_fft,
        hop,
        n_fft,
        window,
        _CENTER,
        _NORMALIZED,
        None,
        None,
        False,
    ), {}


def key_shape(binding):
    return [
        binding["batch"],
        binding["signal_len"],
        binding["n_fft"],
        binding["hop"],
    ]


def config(binding, dtype):
    batch = binding["batch"]
    signal_len = binding["signal_len"]
    n_fft = binding["n_fft"]
    hop = binding["hop"]
    dt = str(dtype)
    frames = signal_len // hop + 1
    return {
        "inputs": {
            "args": [
                {
                    "shape": [batch, n_fft // 2 + 1, frames],
                    "dtype": "torch.complex64",
                    "note": "spectrum from torch.stft",
                },
                {"scalar": n_fft},
                {"scalar": hop},
                {"scalar": n_fft},
                {"shape": [n_fft], "dtype": dt, "note": "hann window"},
                {"scalar": _CENTER},
                {"scalar": _NORMALIZED},
                None,
                None,
                {"scalar": False},
            ],
            "kwargs": {},
        },
        "outputs": {"out": {"shape": [batch, signal_len], "dtype": dt}},
        "dims": {
            "batch": batch,
            "signal_len": signal_len,
            "n_fft": n_fft,
            "hop_length": hop,
            "frames": frames,
        },
    }
