import pytest
import torch

from . import base, consts


class T_Benchmark(base.Benchmark):
    """Benchmark for t_ (inplace transpose) which only applies to 2D tensors."""

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) == 2:
                inp = base.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp,


@pytest.mark.t_
def test_t_():
    bench = T_Benchmark(
        op_name="t_",
        torch_op=torch.ops.aten.t_,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
