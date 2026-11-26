import pytest
import torch

import flag_gems

from . import base


class FFTBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = (
            [((32, 256), None), ((4096, 16, 256), None)]
            if self.op_name == "fft_fft"
            else [
                ((32, 121), 240),
                ((1, 384, 240, 121), 240),
                ((1, 384, 240, 240), 240),
            ]
        )

    def get_input_iter(self, dtype):
        for shape, length in self.shapes:
            yield torch.randn(shape, dtype=dtype, device=self.device), {"n": length}


@pytest.mark.fft_fft
def test_fft_fft_perf():
    bench = FFTBenchmark(
        op_name="fft_fft", torch_op=torch.fft.fft, dtypes=[torch.float32]
    )
    bench.set_gems(flag_gems.fft_fft)
    bench.run()


@pytest.mark.fft_irfft
def test_fft_irfft_perf():
    bench = FFTBenchmark(
        op_name="fft_irfft", torch_op=torch.fft.irfft, dtypes=[torch.complex64]
    )
    bench.set_gems(flag_gems.fft_irfft)
    bench.run()
