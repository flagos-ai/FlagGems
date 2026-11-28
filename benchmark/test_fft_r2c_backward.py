# Copyright 2026 FlagOS Contributors

"""Performance benchmark for ``fft_r2c_backward``.

The reference is the ATen manual backward formula expressed with
``_fft_c2c`` on a zero-padded complex grad, since there is no registered
``aten::_fft_r2c_backward`` op to dispatch through.
"""

import pytest
import torch

import flag_gems

from . import base

CASES = [
    ((9,), 16, -1, 0),
    ((4, 9), 16, -1, 0),
    ((2, 7), 12, -1, 1),
    ((64, 129), 256, -1, 0),
    ((2, 8, 17), 32, 1, 0),
    ((2, 8, 17), 32, -1, 0),
    ((8, 64, 2049), 4096, -1, 0),
    ((512, 2049), 4096, -1, 0),
]


def _reference(grad, last_dim_size=16, dim=-1, normalization=0, onesided=1):
    d = int(dim) % grad.ndim
    N = int(last_dim_size)
    K = grad.size(d)
    full = torch.zeros(
        list(grad.shape[:d]) + [N] + list(grad.shape[d + 1 :]),
        dtype=grad.dtype,
        device=grad.device,
    )
    full.narrow(d, 0, K).copy_(grad)
    return torch.ops.aten._fft_c2c(full, [d], int(normalization), False).real


class FftR2cBackwardBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "operator configuration"

    def set_shapes(self, shape_file_path=None):
        self.shapes = CASES

    def get_input_iter(self, dtype):
        if dtype not in (torch.float32, torch.float64):
            return
        cdt = torch.complex128 if dtype == torch.float64 else torch.complex64
        for case in self.shapes:
            shape, N, dim, norm = case
            grad = torch.randn(shape, dtype=cdt, device=self.device)
            yield grad, {
                "dim": dim,
                "normalization": norm,
                "last_dim_size": N,
                "onesided": 1,
            }

    def get_torch_op(self):
        return _reference


@pytest.mark.fft_r2c_backward
def test_fft_r2c_backward():
    bench = FftR2cBackwardBenchmark(
        op_name="fft_r2c_backward",
        torch_op=_reference,
        # Single dtype: fp32 is the production dtype for this FFT backward
        # gradient; fp64 only affects reference tolerance.
        dtypes=[torch.float32],
    )
    bench.set_gems(flag_gems.fft_r2c_backward)
    bench.run()
