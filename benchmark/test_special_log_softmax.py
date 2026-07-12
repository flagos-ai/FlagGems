import pytest
import torch

from . import base, consts


class SpecialLogSoftmaxBenchmark(base.Benchmark):
    r"""Benchmark for special_log_softmax, reduction over last dim."""

    def set_more_shapes(self):
        # Additional shapes for --level more
        return [(1024, 2**i) for i in range(0, 21, 4)]

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            if len(shape) >= 2:
                inp = base.generate_tensor_input(shape, cur_dtype, self.device)
                yield inp, -1


@pytest.mark.special_log_softmax
def test_special_log_softmax():
    bench = SpecialLogSoftmaxBenchmark(
        op_name="special_log_softmax",
        torch_op=torch.special.log_softmax,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
