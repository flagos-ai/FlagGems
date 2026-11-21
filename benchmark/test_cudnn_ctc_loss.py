import pytest
import torch

import flag_gems

from . import base

# `cudnn_ctc_loss` only accepts float32 `log_probs`.
CTC_DTYPES = [torch.float32]


def cudnn_ctc_loss_input_fn(shape, dtype, device):
    """Build one `cudnn_ctc_loss` invocation for a (T, N, C, S) shape."""
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = torch.log_softmax(raw, dim=-1).to(dtype)

    input_lengths = [t_steps] * batch
    target_lengths = [max_target] * batch
    targets = torch.randint(
        1, classes, (batch * max_target,), dtype=torch.int32, device="cpu"
    )
    yield (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        0,
        False,
        False,
    )


class CudnnCtcLossBenchmark(base.Benchmark):
    """Benchmark `cudnn_ctc_loss` over typical CTC problem shapes."""

    # (T, N, C, S) tuples spanning small/medium/large batch and class counts,
    # matching typical CTC training workloads.
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_shapes(self, shape_file_path=None):
        # A CTC problem is described by a (T, N, C, S) tuple, so the generic
        # shape files are ignored in favour of the tuples above.
        self.shapes = self.DEFAULT_SHAPES
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_shapes(self):
        return []

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from cudnn_ctc_loss_input_fn(shape, dtype, self.device)


@pytest.mark.cudnn_ctc_loss
def test_cudnn_ctc_loss():
    bench = CudnnCtcLossBenchmark(
        op_name="cudnn_ctc_loss",
        torch_op=torch.ops.aten._cudnn_ctc_loss,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems.cudnn_ctc_loss)
    bench.run()
