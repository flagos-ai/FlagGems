# Copyright 2026, The FlagOS Contributors.
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

from flag_gems import istft

from . import base

# (batch, signal length, n_fft, hop_length). A Hann window and the default
# centered/onesided configuration match the common stft -> istft round trip;
# n_fft=1024 covers the large-transform regime.
ISTFT_SHAPES = [
    (1, 4096, 256, 64),
    (8, 4096, 256, 64),
    (32, 4096, 256, 64),
    (8, 16384, 256, 64),
    (8, 16384, 1024, 256),
    (1, 65536, 512, 128),
]


class IstftBenchmark(base.Benchmark):
    """Benchmark for aten::istft (inverse short-time Fourier transform)."""

    DEFAULT_SHAPE_DESC = "batch, signal length, n_fft, hop_length"

    def set_shapes(self, shape_file_path=None):
        self.shapes = ISTFT_SHAPES

    def get_input_iter(self, dtype):
        for batch, signal_len, n_fft, hop in self.shapes:
            window = torch.hann_window(n_fft, dtype=dtype, device=self.device)
            signal = torch.randn(batch, signal_len, dtype=dtype, device=self.device)
            spec = torch.stft(
                signal,
                n_fft,
                hop_length=hop,
                win_length=n_fft,
                window=window,
                center=True,
                return_complex=True,
            )
            # positional: input, n_fft, hop_length, win_length, window,
            # center, normalized, onesided, length, return_complex
            yield spec, n_fft, hop, n_fft, window, True, False, None, None, False


@pytest.mark.istft
def test_istft():
    bench = IstftBenchmark(
        op_name="istft",
        torch_op=torch.istft,
        # cuFFT only accepts complex64/complex128 spectra, so the istft
        # benchmark is limited to float32 (fp16/complex32 is experimental and
        # much slower through torch.stft).
        dtypes=[torch.float32],
    )
    bench.set_gems(istft)
    bench.run()
