import os
import sys

# Ensure flag_gems is imported from the current worktree, not from
# an editable install or sys.path entry pointing to a different worktree.
for _i, _f in enumerate(sys.meta_path):
    if type(_f).__name__ == "ScikitBuildRedirectingFinder":
        del sys.meta_path[_i]
        break
_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, _src)

# conftest.py imports flag_gems before our workaround runs, so the stale
# module from another worktree is cached in sys.modules.  Clear it.
for _mod in list(sys.modules.keys()):
    if _mod == "flag_gems" or _mod.startswith("flag_gems."):
        del sys.modules[_mod]

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base  # noqa: E402

# This operator operates with int32 containing int4 values (a subset of consts.INT_DTYPES).
# Representative (M, N) shapes commonly found in int4-quantized linear layers
CONVERT_WEIGHT_SHAPES = [
    (16, 64),
    (16, 128),
    (32, 128),
    (64, 256),
]


def _convert_weight_input_fn(shape, cur_dtype, device):
    """Input function for convert_weight_to_int4pack benchmark."""
    M, N = shape
    # Input should be int32 for the operation
    x = torch.randint(0, 16, (M, N), dtype=torch.int32, device=device)
    innerKTiles = 2
    yield x, innerKTiles


class ConvertWeightBenchmark(base.Benchmark):
    """Benchmark for convert_weight_to_int4pack operator."""

    def set_shapes(self, shape_file_path=None):
        self.shapes = CONVERT_WEIGHT_SHAPES

    def get_input_iter(self, cur_dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)

    def get_tflops(self, op, *args, **kwargs):
        # TFLOPS doesn't make much sense for this operation
        # Return a placeholder
        return 0


@pytest.mark.convert_weight_to_int4pack
def test_convert_weight_to_int4pack():
    # dtype=torch.int32 is required since the input represents raw int4 weight values
    bench = ConvertWeightBenchmark(
        op_name="convert_weight_to_int4pack",
        torch_op=flag_gems._convert_weight_to_int4pack,
        dtypes=[torch.int32],
    )
    bench.input_fn = _convert_weight_input_fn
    bench.run()
