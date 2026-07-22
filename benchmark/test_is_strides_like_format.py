# benchmark/test_is_strides_like_format.py
import pytest
import torch

from . import base, consts

# Check if pandas is available (required for benchmark profiling on Ascend)
try:
    import pandas  # noqa: F401

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

_FORMAT_MAP = {
    "channels_last": torch.channels_last,
    "channels_last_3d": torch.channels_last_3d,
}


def _torch_is_strides_like_format(x, fmt):
    if fmt not in _FORMAT_MAP:
        return False
    return torch.ops.aten.is_strides_like_format(x, _FORMAT_MAP[fmt])


class IsStridesLikeFormatBenchmark(base.Benchmark):
    def set_more_shapes(self):
        return [
            (2, 3),  # 2D
            (4, 5, 6),  # 3D
            (2, 3, 4, 5),  # 4D
            (8, 3, 224, 224),  # 4D large
            (2, 3, 4, 5, 6),  # 5D
        ]

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            x = torch.randn(shape, dtype=dtype, device=self.device)
            for fmt in ["channels_last", "channels_last_3d"]:
                yield x, fmt


@pytest.mark.is_strides_like_format
@pytest.mark.skipif(
    not HAS_PANDAS,
    reason="pandas not installed, skipping benchmark (Ascend CI environment)",
)
def test_benchmark():
    bench = IsStridesLikeFormatBenchmark(
        op_name="is_strides_like_format",
        torch_op=_torch_is_strides_like_format,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
