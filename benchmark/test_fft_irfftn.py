import pytest
import torch

from benchmark.consts import COMPLEX_DTYPES

from . import base

# COMPLEX_DTYPES are used instead of FLOAT_DTYPES because fft_irfftn
# operates on complex input (half-Hermitian from rfftn).


# FFT irfftn benchmark shapes (only 1D shapes supported)
FFT_IRFFTN_SHAPES = [
    (8,),
    (16,),
    (32,),
    (64,),
    (128,),
    (256,),
    (512,),
    (1024,),
]


class FFTIRFFTNBenchmark(base.GenericBenchmark):
    """Benchmark for 1D fft_irfftn only."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = FFT_IRFFTN_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            # Generate a real input of shape = shape
            real_input = torch.randn(shape, dtype=torch.float32, device=self.device)
            # Get the half-Hermitian complex output from rfftn
            complex_input = torch.fft.rfftn(real_input)
            # Yield as single tensor; default s reconstructs original signal length
            yield (complex_input,)


def fft_irfftn_input_fn(shape, dtype, device):
    """Generate input for fft_irfftn benchmark.

    The input should be the half-Hermitian output from rfftn.
    """
    # Generate a real input of shape = shape
    real_input = torch.randn(shape, dtype=torch.float32, device=device)
    # Get the half-Hermitian complex output from rfftn
    complex_input = torch.fft.rfftn(real_input)
    yield (complex_input,)


@pytest.mark.fft_irfftn
def test_fft_irfftn():
    bench = FFTIRFFTNBenchmark(
        input_fn=fft_irfftn_input_fn,
        op_name="fft_irfftn",
        torch_op=torch.fft.irfftn,
        dtypes=COMPLEX_DTYPES,
    )
    bench.run()
