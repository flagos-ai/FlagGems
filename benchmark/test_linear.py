import pytest
import torch

from . import base, consts


class LinearBenchmark(base.BlasBenchmark):
    def get_tflops(self, op, *args, **kwargs):
        input_tensor = args[0]
        weight = args[1]
        batch = input_tensor.shape[0]
        out_features = weight.shape[0]
        in_features = weight.shape[1]
        return batch * out_features * (2 * in_features + 1)


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    input_tensor = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([n, k], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.linear
def test_linear():
    bench = LinearBenchmark(
        op_name="linear",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.linear,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
