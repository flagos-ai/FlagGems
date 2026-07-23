import sys

# The concurrent agent runner may leave a stale editable install redirector pointing to
# a different worktree. Remove it so our worktree's flag_gems is loaded.
sys.meta_path = [
    m for m in sys.meta_path if "ScikitBuildRedirectingFinder" not in str(type(m))
]

# Swap our worktree's src to the front of sys.path.
_our_src = "/tmp/flaggems_agent_worktrees/agent__nested_view_from_buffer_copy_179605-1781252494/src"
sys.path = [_our_src] + [p for p in sys.path if p != _our_src]

# Clear any cached flag_gems modules.
for _k in list(sys.modules):
    if "flag_gems" in _k:
        del sys.modules[_k]

import pytest  # noqa: E402
import torch  # noqa: E402

import flag_gems  # noqa: E402

from . import base, consts  # noqa: E402


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
