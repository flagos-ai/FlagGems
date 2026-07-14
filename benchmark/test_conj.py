import pytest
import torch

import flag_gems

from . import base, consts

def _input_fn(shape, dtype, device):
    if dtype.is_complex:
        float_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
        real = torch.randn(shape, dtype=float_dtype, device=device)
        imag = torch.randn(shape, dtype=float_dtype, device=device)
        input_tensor = torch.complex(real, imag).to(dtype)
    elif dtype.is_floating_point:
        input_tensor = torch.randn(shape, dtype=dtype, device=device)
    else:
        input_tensor = torch.randn(shape, device=device).to(dtype)
    yield (input_tensor,)

class ConjBenchmark(base.GenericBenchmarkExcluse3D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def set_shapes(self, shape_file_path=None):
        conj_shapes = [
            (256,),
            (2048, 2048),
            (128, 512, 256),
            (32, 64),
            (512, 1024),
            (2, 3, 4),
        ]
        self.shapes = conj_shapes

    def set_more_shapes(self):
        return None

@pytest.mark.conj
def test_conj():
    dtypes = consts.FLOAT_DTYPES + consts.COMPLEX_DTYPES

    bench = ConjBenchmark(
        input_fn=_input_fn,
        op_name="conj",
        torch_op=torch.conj,
        dtypes=dtypes,
    )

    bench.set_gems(flag_gems.conj)
    bench.run()
