import pytest
import torch

from . import attri_util as attr_utils
from . import performance_utils as utils


class SliceBackwardBenchmark(utils.GenericBenchmark):
    def set_more_shapes(self):
        SLICE_BACKWARD_SHAPES = (
            (128, 256),
            (1024, 1024),
            (512, 1024, 512),
            (16, 8192, 4096),
            (8, 4096, 11008),
            (4, 32, 4096, 128),
            (32, 256, 256, 128),
        )

        self.shapes = SLICE_BACKWARD_SHAPES
        return None


def _get_gbps(args, latency):
    grad_output, shape, dim, start, end, step = args

    bytes_per_element = grad_output.element_size()

    output_numel = 1
    for s in shape:
        output_numel *= s

    total_bytes = (grad_output.numel() + output_numel) * bytes_per_element

    return total_bytes / latency / 1e9


def _input_fn(shape, dtype, device):
    dim = 0 if len(shape) == 1 else 1

    start = 0
    end = shape[dim]
    step = 2

    size = shape[dim]

    start = start % size
    end = end % (size + 1)

    if end < start:
        end, start = start, end
    elif end == start:
        end = size

    slice_len = (end - start + step - 1) // step

    valid_shape = list(shape)
    valid_shape[dim] = slice_len

    grad_output = torch.randn(
        valid_shape,
        dtype=dtype,
        device=device,
    )

    yield grad_output, shape, dim, start, end, step


@pytest.mark.slice_backward
def test_slice_backward():
    bench = SliceBackwardBenchmark(
        op_name="slice_backward",
        torch_op=torch.ops.aten.slice_backward,
        input_fn=_input_fn,
        dtypes=attr_utils.FLOAT_DTYPES,
        get_gbps=_get_gbps,
    )

    bench.run()
