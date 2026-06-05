import pytest
import torch

from . import base, consts


class LinearBenchmark(base.BlasBenchmark):
    """
    benchmark for linear
    """

    def get_tflops(self, op, *args, **kwargs):
        if self.op_name == "linear":
            # linear: y = x @ W^T + b
            # total_flops = batch * out_features * (2 * in_features + 1)
            input_tensor = args[0]
            weight = args[1]
            batch = input_tensor.shape[0]
            out_features = weight.shape[0]
            in_features = weight.shape[1]
            return batch * out_features * (2 * in_features + 1)
        return super().get_tflops(op, *args, **kwargs)


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    """
    Input function for linear benchmark.
    - input: (m, k) where m is batch size, k is in_features
    - weight: (n, k) where n is out_features
    - bias: (n,)
    """
    # Here m is actually batch size, k is in_features, n is out_features
    input_tensor = torch.randn([m, k], dtype=dtype, device=device)
    weight = torch.randn([n, k], dtype=dtype, device=device)
    bias = torch.randn([n], dtype=dtype, device=device)
    yield input_tensor, weight, bias


@pytest.mark.linear
def test_linear(monkeypatch):
    bench = LinearBenchmark(
        op_name="linear",
        input_fn=_input_fn,
        torch_op=torch.nn.functional.linear,
        dtypes=consts.FLOAT_DTYPES,
    )

    bench.run()
