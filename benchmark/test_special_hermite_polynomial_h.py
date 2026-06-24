import pytest
import torch

from . import base


@pytest.mark.special_hermite_polynomial_h
def test_special_hermite_polynomial_h():
    class _HermiteBenchmark(base.BinaryPointwiseBenchmark):
        def get_input_iter(self, dtype):
            for shape in self.shapes:
                inp1 = base.generate_tensor_input(shape, dtype, self.device)
                # n must be in [0, 9] per operator validation
                inp2 = torch.randint(0, 10, shape, device=self.device).to(dtype)
                yield inp1, inp2

    bench = _HermiteBenchmark(
        op_name="special_hermite_polynomial_h",
        torch_op=torch.special.hermite_polynomial_h,
        # special.* ops only support float32
        dtypes=[torch.float32],
    )
    bench.run()
