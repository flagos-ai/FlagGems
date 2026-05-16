import pytest
import torch

from . import base, consts


class SelectBackwardBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        SELECT_BACKWARD_SHAPES = (
            (128, 256),
            (1024, 1024),
            (512, 1024, 512),
            (16, 8192, 4096),
            (8, 4096, 11008),
            (4, 32, 4096, 128),
            (32, 256, 256, 128),
        )

        self.shapes = SELECT_BACKWARD_SHAPES
        return None


def _get_gbps(args, latency):
    grad_output, input_sizes, dim, index = args

    bytes_per_element = grad_output.element_size()

    output_numel = 1
    for s in input_sizes:
        output_numel *= s

    total_bytes = (grad_output.numel() + output_numel) * bytes_per_element

    return total_bytes / latency / 1e9


def _input_fn(shape, dtype, device):
    ndim = len(shape)

    for dim in [0, ndim // 2, -1]:
        actual_dim = dim + ndim if dim < 0 else dim

        if actual_dim >= ndim:
            continue

        dim_size = shape[actual_dim]
        index = dim_size // 2

        grad_shape = list(shape)
        grad_shape.pop(actual_dim)

        grad_output = torch.randn(
            grad_shape,
            dtype=dtype,
            device=device,
        )

        yield grad_output, shape, actual_dim, index


@pytest.mark.select_backward
def test_select_backward():
    bench = SelectBackwardBenchmark(
        op_name="select_backward",
        torch_op=torch.ops.aten.select_backward,
        input_fn=_input_fn,
        dtypes=consts.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
