import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts


def deep_parse_shape(item):
    if isinstance(item, torch.Tensor):
        return item.size()
    if isinstance(item, (int, float, str)):
        return item
    if isinstance(item, (list, tuple)):
        return [deep_parse_shape(sub) for sub in item]
    return None


# Padded -> nested benchmark shapes: (batch_size, padded_len, head_dim).
# head_dim kept at 64/128 so a component row spans one or a few kernel blocks;
# padded_len mirrors the sequence lengths seen around transformer attention.
NESTED_FROM_PADDED_SHAPES = [
    (8, 8, 64),
    (16, 16, 64),
    (32, 32, 64),
    (64, 64, 64),
    (128, 64, 64),
    (256, 128, 64),
    (512, 256, 128),
]


class NestedFromPaddedBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_FROM_PADDED_SHAPES

    def record_shapes(self, *args, **kwargs):
        # The result is a nested tensor; NestedTensorImpl does not support
        # .size(), so record the padded input shape instead of the base
        # implementation which would raise.
        return [args[0].size()] + [deep_parse_shape(a) for a in args[1:]]

    def get_input_iter(self, cur_dtype):
        for batch_size, max_len, head_dim in self.shapes:
            # Random per-batch sequence lengths in [1, max_len], with component 0
            # pinned to the full padded length so the layout is deterministic.
            np.random.seed(42)
            lengths = np.random.randint(1, max_len + 1, size=batch_size)
            lengths[0] = max_len
            padded = torch.randn(
                (batch_size, max_len, head_dim), dtype=cur_dtype, device=self.device
            )
            sizes = torch.stack(
                [
                    torch.from_numpy(lengths).to(torch.int64),
                    torch.full((batch_size,), head_dim, dtype=torch.int64),
                ],
                dim=1,
            )
            yield padded, sizes


@pytest.mark.nested_from_padded
def test_nested_from_padded():
    # gems_op is passed explicitly so the measured latency is the operator
    # itself rather than the GEMS dispatch layer around it.
    bench = NestedFromPaddedBenchmark(
        op_name="nested_from_padded",
        torch_op=torch.ops.aten._nested_from_padded,
        gems_op=flag_gems._nested_from_padded,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
