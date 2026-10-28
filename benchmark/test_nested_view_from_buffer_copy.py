import pytest
import torch

import flag_gems

from . import base, consts

# The native XPU implementation requires host-resident metadata and an exact
# packed buffer (buffer size == sum of component sizes), so the three shapes
# keep their original payload scale and cover the whole buffer with 1:2:3
# component splits.
NESTED_VIEW_SHAPES = [
    (100000, [[16667], [33333], [50000]], [0, 16667, 50000]),
    (50000, [[8333], [16667], [25000]], [0, 8333, 25000]),
    (200000, [[33333], [66667], [100000]], [0, 33333, 100000]),
]


def _native_nested_view(buffer, sizes, strides, offsets):
    """Native ATen baseline, measured exactly like the FlagGems side."""
    native = getattr(torch.ops.aten, "_nested_view_from_buffer_copy").default
    return native(buffer, sizes, strides, offsets)


def _gems_nested_view(buffer, sizes, strides, offsets):
    """FlagGems implementation, same host metadata as the native baseline."""
    return flag_gems._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)


class NestedViewFromBufferCopyBenchmark(base.Benchmark):
    """Benchmark native ATen and the Kunlunxin implementation separately."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_VIEW_SHAPES

    def get_input_iter(self, cur_dtype):
        for buffer_size, sizes, offsets in self.shapes:
            assert sum(size[0] for size in sizes) == buffer_size
            buffer = torch.randn(
                buffer_size,
                dtype=cur_dtype,
                device=self.device,
            )
            strides = [[1] for _ in sizes]
            sizes = torch.tensor(sizes, dtype=torch.int64)
            strides = torch.tensor(strides, dtype=torch.int64)
            offsets = torch.tensor(offsets, dtype=torch.int64)
            yield buffer, sizes, strides, offsets

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_buffer_copy
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_view_from_buffer_copy(dtype):
    bench = NestedViewFromBufferCopyBenchmark(
        op_name="nested_view_from_buffer_copy",
        torch_op=_native_nested_view,
        gems_op=_gems_nested_view,
        dtypes=[dtype],
    )
    bench.run()
