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

from typing import Generator

import pytest
import torch

from . import base, consts

ISTFT_REAL_DTYPES = [dtype for dtype in consts.FLOAT_DTYPES if dtype == torch.float32]


class ISTFTBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        # Batch, transform length, frame count, and hop length cover common audio sizes.
        self.shapes = [
            (1, 64, 16, 16),
            (2, 256, 24, 64),
            (4, 512, 32, 128),
        ]
        self.shape_desc = "BATCH, N_FFT, N_FRAMES, HOP_LENGTH"

    def get_input_iter(self, cur_dtype) -> Generator:
        for batch, n_fft, n_frames, hop_length in self.shapes:
            n_freq = n_fft // 2 + 1
            real = torch.randn(
                (batch, n_freq, n_frames),
                dtype=ISTFT_REAL_DTYPES[0],
                device=self.device,
            )
            imag = torch.randn_like(real)
            inp = torch.complex(real, imag)
            window = torch.hann_window(n_fft, device=self.device)
            yield inp, n_fft, hop_length, n_fft, window

    def get_tflops(self, op, *args, **kwargs):
        inp, n_fft = args[:2]
        n_transforms = inp.shape[0] * inp.shape[-1]
        return n_transforms * 5 * n_fft * (n_fft.bit_length() - 1)


@pytest.mark.istft
def test_istft():
    bench = ISTFTBenchmark(
        op_name="istft",
        torch_op=torch.istft,
        # The Triton FFT path intentionally supports complex64 only.
        dtypes=consts.COMPLEX_DTYPES,
    )
    bench.run()
