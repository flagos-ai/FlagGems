import pytest
import torch

import flag_gems

from . import base, consts


def _native_nested_view(buffer, sizes, strides, offsets):
    """Native ATen baseline, measured exactly like the FlagGems side."""
    return torch.ops.aten._nested_view_from_buffer.default(
        buffer, sizes, strides, offsets
    )


def _gems_nested_view(buffer, sizes, strides, offsets):
    """FlagGems implementation, same host metadata as the native baseline."""
    return flag_gems._nested_view_from_buffer(buffer, sizes, strides, offsets)


# Metadata must be host-resident: the native CUDA kernel reads it through CPU
# raw pointers (device-resident metadata segfaults; see the implementation
# note in src/flag_gems/ops/_nested_view_from_buffer.py). The shapes keep the
# payload scale of the sibling _nested_view_from_buffer_copy benchmark with
# 2-D components; the buffer is exactly packed (offsets are the running
# element offsets, sum of component numels == buffer_size).
NESTED_VIEW_SHAPES = [
    (
        90000,
        [[5000, 3], [10000, 3], [15000, 3]],
        [[3, 1], [3, 1], [3, 1]],
        [0, 15000, 45000],
    ),
    (
        270000,
        [[15000, 3], [30000, 3], [45000, 3]],
        [[3, 1], [3, 1], [3, 1]],
        [0, 45000, 135000],
    ),
    (
        810000,
        [[45000, 3], [90000, 3], [135000, 3]],
        [[3, 1], [3, 1], [3, 1]],
        [0, 135000, 405000],
    ),
]


class NestedViewFromBufferBenchmark(base.Benchmark):
    """Benchmark the metadata-placement repair overhead of the wrapper.

    The op itself is O(1) metadata work (the result aliases the buffer), so
    the measured difference against the native baseline is the wrapper's
    metadata staging logic plus the dispatch hop.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_VIEW_SHAPES

    def get_input_iter(self, cur_dtype):
        for buffer_size, sizes, strides, offsets in self.shapes:
            assert sum(size[0] * size[1] for size in sizes) <= buffer_size
            buffer = torch.randn(buffer_size, dtype=cur_dtype, device=self.device)
            sizes_t = torch.tensor(sizes, dtype=torch.int64)
            strides_t = torch.tensor(strides, dtype=torch.int64)
            offsets_t = torch.tensor(offsets, dtype=torch.int64)
            yield buffer, sizes_t, strides_t, offsets_t

    def get_gbps(self, args, latency=None):
        # Pure view op: the result aliases the buffer, no data movement.
        return 0.0

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_buffer
@pytest.mark.nested_view_from_buffer
@pytest.mark.parametrize("dtype", consts.FLOAT_DTYPES)
def test_nested_view_from_buffer(dtype):
    bench = NestedViewFromBufferBenchmark(
        op_name="nested_view_from_buffer",
        torch_op=_native_nested_view,
        gems_op=_gems_nested_view,
        dtypes=[dtype],
    )
    bench.run()
