import pytest
import torch

import flag_gems

from .base import Benchmark


class FFTBenchmark(Benchmark):
    def set_shapes(self, shape_file_path=None):
        # FFT requires N to be power-of-two <= 1024
        self.shapes = [
            (128, 64),
            (128, 128),
            (128, 256),
            (128, 512),
            (128, 1024),
            (256, 64),
            (256, 128),
            (256, 256),
            (256, 512),
            (256, 1024),
            (512, 64),
            (512, 128),
            (512, 256),
            (512, 512),
            (512, 1024),
            (1024, 64),
            (1024, 128),
            (1024, 256),
            (1024, 512),
            (1024, 1024),
            (4096, 64),
            (4096, 128),
            (4096, 256),
            (4096, 512),
            (4096, 1024),
        ]

    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            m, n = shape
            real = torch.randn((m, n), device=self.device, dtype=torch.float32)
            imag = torch.randn((m, n), device=self.device, dtype=torch.float32)
            x = torch.complex(real, imag)
            yield x,

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        m, n = x.shape
        # FFT is O(N log N) per row
        flops_per_fft = 5 * n * (n.bit_length() - 1)  # 5 ops per butterfly
        return m * flops_per_fft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.fft
def test_perf_fft():
    def torch_fft(x):
        return torch.fft.fft(x)

    def gems_fft(x):
        return flag_gems.ops.fft(x)

    bench = FFTBenchmark(
        op_name="fft",
        torch_op=torch_fft,
        dtypes=[torch.complex64],
    )
    bench.set_gems(gems_fft)
    bench.run()
