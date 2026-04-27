import pytest
import torch

from . import attri_util as attrs
from . import performance_utils as base


class BaddbmmBenchmark(base.BlasBenchmark):
    def set_more_shapes(self):
        model_shapes_list = attrs.model_shapes()

        skip_shapes = [
            (4, 8192, 128256, 4096),
            (4, 8192, 152064, 3584),
        ]

        filtered = []
        for shape in model_shapes_list:
            if shape not in skip_shapes:
                filtered.append(shape)

        return filtered

    def get_tflops(self, op, *args, **kwargs):
        # shape(b,m,k)(b,k,n)
        # total_flops = b * m * n * (2 * k + 1)
        total_flops = (
            args[1].shape[0]
            * args[1].shape[1]
            * args[2].shape[2]
            * (args[1].shape[2] * 2 + 1)
        )
        return total_flops


def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([b, m, k], dtype=dtype, device=device, requires_grad=True)

    if b_column_major:
        inp2 = torch.randn([b, n, k], dtype=dtype, device=device, requires_grad=True)
        inp2 = inp2.transpose(1, 2).contiguous()
    else:
        inp2 = torch.randn([b, k, n], dtype=dtype, device=device, requires_grad=True)

    bias = torch.randn([b, m, n], dtype=dtype, device=device, requires_grad=True)

    yield bias, inp1, inp2


@pytest.mark.baddbmm
def test_baddbmm():
    bench = BaddbmmBenchmark(
        op_name="baddbmm",
        input_fn=_input_fn,
        torch_op=torch.baddbmm,
        dtypes=attrs.FLOAT_DTYPES,
    )

    bench.run()
