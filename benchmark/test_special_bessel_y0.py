import pytest
import torch

import flag_gems

from . import base


@pytest.mark.skipif(
    flag_gems.vendor_name == "enflame", reason="enflame dose not support fp64"
)
@pytest.mark.special_bessel_y0
def test_special_bessel_y0():
    class BesselY0Benchmark(base.UnaryPointwiseBenchmark):
        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                if cur_dtype == torch.float64:
                    inp = torch.randn(shape, dtype=torch.float64, device=self.device)
                else:
                    inp = base.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,

    bench = BesselY0Benchmark(
        op_name="special_bessel_y0",
        torch_op=torch.special.bessel_y0,
        # torch.special.bessel_y0 supports float32 and float64
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
