import pytest
import torch

import flag_gems

from . import base, consts


class ExpandBackwardBenchmark(base.Benchmark):
    # The backward keeps the input, the expanded gradient and the reduced
    # gradient live at once. The 2**30-element "from perf" default shapes
    # would expand 8x to 32 GiB per tensor, so the sweep uses shapes that stay
    # in the few-GiB range while still covering the three reduction layouts
    # (prepended leading dim, broadcast middle dim, broadcast trailing dim).
    SHAPES = [
        (64, 64),
        (1024, 1024),
        (64, 512, 512),
    ]

    def set_shapes(self, shape_file_path=None):
        # Bypass the shared shape YAML: ``set_shapes`` walks the class MRO and
        # would otherwise pick up the catch-all ``Benchmark:`` entry, which
        # carries the billion-element shapes this op cannot afford.
        self.shapes = list(self.SHAPES)

    def get_input_iter(self, dtype):
        """Yield ``(input, size)`` pairs for an expand forward.

        ``shape`` is the forward input shape; the expanded gradient shape is
        derived by broadcasting every size-1 dim out to 8 (and prepending one
        leading dim), which exercises the three reduction layouts of
        ``ExpandBackward0``: prepended leading dims, a broadcast middle dim and
        a broadcast trailing dim.
        """
        for shape in self.shapes:
            inp = torch.randn(
                shape, dtype=dtype, device=flag_gems.device, requires_grad=True
            )
            grad_shape = list(shape)
            grad_shape = [8] + grad_shape
            grad_shape = [s if s != 1 else 8 for s in grad_shape]
            yield inp, grad_shape


@pytest.mark.expand_backward
def test_expand_backward():
    # ``aten::expand`` keeps its own ExpandBackward0 node (a sum_to_size), so
    # drive the gradient through the FlagGems autograd bridge explicitly --
    # otherwise this would time the native sum chain, not this kernel.
    from flag_gems.ops.expand_backward import ExpandFunction

    bench = ExpandBackwardBenchmark(
        op_name="expand_backward",
        torch_op=lambda inp, size: inp.expand(tuple(size)),
        gems_op=ExpandFunction.apply,
        dtypes=consts.FLOAT_DTYPES,
        is_backward=True,
    )
    bench.run()
