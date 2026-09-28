import pytest
import torch

import flag_gems

from . import base, consts


class NestedViewFromBufferCopyBenchmark(base.Benchmark):
    """
    Benchmark for _nested_view_from_buffer_copy operator.
    """

    def set_shapes(self, shape_file_path=None):
        # Three buffer size configurations covering small/medium/large nested tensor cases
        self.shapes = [
            (100000, [[1000], [2000], [3000]], [[1], [1], [1]], [0, 1000, 3000]),
            (50000, [[500], [1000], [1500]], [[1], [1], [1]], [0, 500, 1500]),
            (200000, [[5000], [10000], [15000]], [[1], [1], [1]], [0, 5000, 15000]),
        ]

    def get_input_iter(self, cur_dtype):
        for buffer_size, sizes, strides, offsets in self.shapes:
            buffer = torch.randn(buffer_size, dtype=cur_dtype, device=self.device)
            sizes_t = torch.tensor(sizes, dtype=torch.int64, device=self.device)
            strides_t = torch.tensor(strides, dtype=torch.int64, device=self.device)
            offsets_t = torch.tensor(offsets, dtype=torch.int64, device=self.device)
            yield buffer, sizes_t, strides_t, offsets_t

    def get_tflops(self, op, *args, **kwargs):
        return 0.0


@pytest.mark.nested_view_from_buffer_copy
@pytest.mark.parametrize(
    "dtype",
    consts.FLOAT_DTYPES,
)
def test_nested_view_from_buffer_copy(dtype):
    bench = NestedViewFromBufferCopyBenchmark(
        op_name="nested_view_from_buffer_copy",
        torch_op=flag_gems._nested_view_from_buffer_copy,
        dtypes=[dtype],
    )
    bench.run()
