import numpy as np
import pytest
import torch

import flag_gems

from . import base, consts


def _gems_op(padded, example):
    return flag_gems._nested_from_padded_and_nested_example(padded, example)


def _native_op(padded, example):
    return torch.ops.aten._nested_from_padded_and_nested_example.default(
        padded, example
    )


# (B, T, D) padded shapes spanning small-to-large batch/length/feature so the
# un-padding kernel is exercised across a range of inner-dim widths, matching
# the sibling nested_from_padded_tensor benchmark.
NESTED_FROM_PADDED_AND_NESTED_EXAMPLE_SHAPES = [
    (8, 16, 16),
    (32, 32, 64),
    (128, 64, 256),
    (512, 128, 64),
]


class NestedFromPaddedAndNestedExampleBenchmark(base.Benchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = NESTED_FROM_PADDED_AND_NESTED_EXAMPLE_SHAPES

    def record_shapes(self, *args, **kwargs):
        # Legacy (strided-layout) nested tensors raise from ``.size()``; report
        # the per-component sizes via ``unbind`` instead so shape recording
        # does not fail on the example input.
        def deep_parse(item):
            if isinstance(item, torch.Tensor) and item.is_nested:
                return [c.size() for c in item.unbind()]
            if isinstance(item, torch.Tensor):
                return item.size()
            if isinstance(item, (int, float, str, torch.dtype)):
                return item
            if isinstance(item, (list, tuple)):
                return [deep_parse(sub_item) for sub_item in item]
            if isinstance(item, dict):
                return {key: deep_parse(sub) for key, sub in item.items()}
            return None

        parsed_args = [deep_parse(arg) for arg in args]
        parsed_kwargs = {key: deep_parse(val) for key, val in kwargs.items()}
        if parsed_args and parsed_kwargs:
            return parsed_args, parsed_kwargs
        return parsed_args if parsed_args else parsed_kwargs

    def get_input_iter(self, cur_dtype):
        for B, T, D in self.shapes:
            np.random.seed(42)
            lengths = np.random.randint(1, T + 1, size=B)
            padded = torch.randn(B, T, D, dtype=cur_dtype, device=self.device)
            components = [padded[b, : int(l)] for b, l in enumerate(lengths)]
            example = torch.nested.nested_tensor(components, device=self.device)
            yield padded, example


@pytest.mark.nested_from_padded_and_nested_example
def test_nested_from_padded_and_nested_example():
    bench = NestedFromPaddedAndNestedExampleBenchmark(
        op_name="nested_from_padded_and_nested_example",
        torch_op=_native_op,
        gems_op=_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
