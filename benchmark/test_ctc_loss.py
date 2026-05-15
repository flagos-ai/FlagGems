import pytest
import torch

import flag_gems

from . import base, consts

CTC_DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def _make_targets(batch, max_target, classes, device, target_layout):
    target_lengths = torch.empty(batch, device=device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=device, dtype=torch.long)
    pieces = []
    for row in range(batch):
        length = max(1, max_target - (row % 5))
        target_lengths[row] = length
        values = (torch.arange(length, device=device, dtype=torch.long) + row) % (
            classes - 1
        )
        values = values + 1
        padded[row, :length] = values
        pieces.append(values)

    if target_layout == "padded":
        targets = padded
    else:
        targets = torch.cat(pieces)
    return targets, target_lengths


def ctc_loss_input_fn(shape, dtype, device):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)

    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, "padded"
    )
    yield (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": 0, "reduction": 1, "zero_infinity": False},
    )

    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, "concatenated"
    )
    yield (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": 0, "reduction": 1, "zero_infinity": False},
    )

    if base.Config.bench_level.value == consts.BenchLevel.COMPREHENSIVE.value:
        targets, target_lengths = _make_targets(
            batch, max_target, classes, device, "padded"
        )
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": 2, "zero_infinity": False},
        )


class CtcLossBenchmark(base.GenericBenchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"

    def set_more_shapes(self):
        return []


@pytest.mark.ctc_loss
def test_ctc_loss():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        input_fn=ctc_loss_input_fn,
        torch_op=torch.ops.aten.ctc_loss.Tensor,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems.ctc_loss)
    bench.run()


@pytest.mark.ctc_loss
def test_ctc_loss_backward():
    bench = CtcLossBenchmark(
        op_name="ctc_loss",
        input_fn=ctc_loss_input_fn,
        torch_op=torch.ops.aten.ctc_loss.Tensor,
        dtypes=CTC_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.ctc_loss)
    bench.run()
