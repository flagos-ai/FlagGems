import pytest
import torch

from . import base


@pytest.mark.special_bessel_j0
def test_special_bessel_j0():
    class BesselJ0Benchmark(base.UnaryPointwiseBenchmark):
        def get_input_iter(self, cur_dtype):
            for shape in self.shapes:
                if cur_dtype == torch.float64:
                    inp = torch.randn(shape, dtype=torch.float64, device=self.device)
                else:
                    inp = base.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,

    bench = BesselJ0Benchmark(
        op_name="special_bessel_j0",
        torch_op=torch.special.bessel_j0,
        # torch.special.bessel_j0 supports float32 and float64
        dtypes=[torch.float32, torch.float64],
    )
    bench.run()
