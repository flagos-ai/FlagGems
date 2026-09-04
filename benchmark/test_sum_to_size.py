from typing import Generator

import pytest

from . import base, consts


class SumToSizeBenchmark(base.UnaryReductionBenchmark):
    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            inp = base.generate_tensor_input(shape, cur_dtype, self.device)
            if inp.ndim > 1:
                # Reverse-broadcast the leading dimension: reduce dim 0.
                size = (1,) + tuple(shape[1:])
            else:
                size = (1,)
            yield inp, size


@pytest.mark.sum_to_size
def test_sum_to_size():
    bench = SumToSizeBenchmark(
        op_name="sum_to_size",
        torch_op=lambda inp, size: inp.sum_to_size(size),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
