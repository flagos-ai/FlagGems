import pytest
import torch

from . import base


@pytest.mark.special_scaled_modified_bessel_k1
def test_special_scaled_modified_bessel_k1():
    bench = base.UnaryPointwiseBenchmark(
        op_name="special_scaled_modified_bessel_k1",
        torch_op=torch.special.scaled_modified_bessel_k1,
        # Bessel K1 not supported for Half/BFloat16 in PyTorch
        dtypes=[torch.float32],
    )
    bench.run()


class _ScaledModifiedBesselK1OutBenchmark(base.UnaryPointwiseOutBenchmark):
    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            out = torch.empty_like(inp)
            yield inp, {"out": out}


@pytest.mark.special_scaled_modified_bessel_k1_out
def test_special_scaled_modified_bessel_k1_out():
    bench = _ScaledModifiedBesselK1OutBenchmark(
        op_name="special_scaled_modified_bessel_k1_out",
        torch_op=torch.special.scaled_modified_bessel_k1,
        # Bessel K1 not supported for Half/BFloat16 in PyTorch
        dtypes=[torch.float32],
    )
    bench.run()
