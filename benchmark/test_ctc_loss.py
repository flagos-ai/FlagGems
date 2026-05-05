import pytest
import torch
import torch.nn.functional as F

import flag_gems

from . import base, consts

CTC_DTYPES = [torch.float32, torch.float16]
if flag_gems.runtime.device.support_bf16:
    CTC_DTYPES.append(torch.bfloat16)


def _reference_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *args,
    **kwargs,
):
    work = log_probs if log_probs.dtype == torch.float32 else log_probs.float()
    out = F.ctc_loss(
        work, targets, input_lengths, target_lengths, *args, **kwargs
    )
    return out.to(log_probs.dtype)


def _make_targets(batch, max_target, classes, device, *, concatenated):
    target_lengths = torch.empty(batch, device=device, dtype=torch.long)
    padded = torch.zeros(batch, max_target, device=device, dtype=torch.long)
    pieces = []
    for row in range(batch):
        length = max(1, max_target - (row % 5))
        target_lengths[row] = length
        values = (
            torch.arange(length, device=device, dtype=torch.long) + row
        ) % (classes - 1)
        values = values + 1
        padded[row, :length] = values
        pieces.append(values)
    if concatenated:
        return torch.cat(pieces), target_lengths
    return padded, target_lengths


def ctc_loss_input_fn(shape, dtype, device):
    t_steps, batch, classes, max_target = shape
    raw = torch.randn(
        t_steps, batch, classes, dtype=torch.float32, device=device
    )
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full(
        (batch,), t_steps, dtype=torch.long, device=device
    )

    for concatenated in (False, True):
        targets, target_lengths = _make_targets(
            batch, max_target, classes, device, concatenated=concatenated
        )
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": "mean", "zero_infinity": False},
        )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        targets, target_lengths = _make_targets(
            batch, max_target, classes, device, concatenated=False
        )
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": "sum", "zero_infinity": False},
        )


class CtcLossBenchmark(base.Benchmark):
    DEFAULT_SHAPES = [
        (64, 4, 32, 16),
        (256, 16, 64, 48),
        (512, 32, 64, 48),
        (1024, 32, 128, 96),
    ]
    DEFAULT_SHAPE_DESC = "T, N, C, S"
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        tensors = [arg for arg in args if torch.is_tensor(arg)]
        total_bytes = sum(t.numel() * t.element_size() for t in tensors)
        return total_bytes * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            yield from self.input_fn(shape, dtype, self.device)


@pytest.mark.ctc_loss
def test_ctc_loss():
    bench = CtcLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_DTYPES,
    )
    bench.set_gems(flag_gems.ctc_loss)
    bench.run()


@pytest.mark.ctc_loss
def test_ctc_loss_backward():
    bench = CtcLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_DTYPES,
        is_backward=True,
    )
    bench.set_gems(flag_gems.ctc_loss)
    bench.run()
