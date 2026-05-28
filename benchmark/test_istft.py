import pytest
import torch

import flag_gems

from .base import Benchmark


class ISTFTBenchmark(Benchmark):
    def set_more_shapes(self):
        return None

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            n_frames, n_fft, n_channels = shape
            real = torch.randn(
                (n_frames, n_fft // 2 + 1, n_channels),
                device=self.device,
                dtype=torch.float32,
            )
            imag = torch.randn(
                (n_frames, n_fft // 2 + 1, n_channels),
                device=self.device,
                dtype=torch.float32,
            )
            x = torch.complex(real, imag)
            window = torch.hann_window(n_fft, device=self.device, dtype=torch.float32)
            yield x, n_fft, None, None, window

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        n_frames, n_freq, n_channels = x.shape
        n_fft = args[1]
        flops_per_ifft = 5 * n_fft * (n_fft.bit_length() - 1)
        flops_per_overlap_add = n_fft
        return n_frames * (flops_per_ifft + flops_per_overlap_add)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.istft
def test_perf_istft():
    def torch_istft(x, n_fft, hop_length, win_length, window):
        return torch.istft(
            x, n_fft, hop_length, win_length, window, return_complex=False
        )

    def gems_istft(x, n_fft, hop_length, win_length, window):
        return flag_gems.ops.istft(
            x, n_fft, hop_length, win_length, window, return_complex=False
        )

    bench = ISTFTBenchmark(
        op_name="istft",
        torch_op=torch_istft,
        dtypes=[torch.complex64],
    )
    bench.set_gems(gems_istft)
    bench.run()
