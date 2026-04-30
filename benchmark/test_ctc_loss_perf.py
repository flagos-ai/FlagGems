from typing import Generator

import pytest
import torch

import flag_gems

from . import base, consts


def _make_seed(shape, *, salt):
    seed = 17 + salt
    for dim in shape:
        seed = seed * 131 + int(dim)
    return seed


def _make_generator(device, shape, *, salt):
    generator = torch.Generator(device=device)
    generator.manual_seed(_make_seed(shape, salt=salt))
    return generator


def _make_targets(
    batch_size, max_target_length, num_classes, device, blank, *, generator
):
    target_lengths = torch.randint(
        1,
        max_target_length + 1,
        (batch_size,),
        device=device,
        dtype=torch.long,
        generator=generator,
    )
    padded = torch.empty(
        (batch_size, max_target_length), device=device, dtype=torch.long
    )
    labels = torch.randint(
        0,
        num_classes - 1,
        (batch_size, max_target_length),
        device=device,
        generator=generator,
    )
    padded.copy_(labels + (labels >= blank).to(labels.dtype))
    return padded, target_lengths


def ctc_loss_input_fn(shape, cur_dtype, device):
    T, N, C, S = shape
    blank = 0
    logits_generator = _make_generator(device, shape, salt=0)
    targets_generator = _make_generator(device, shape, salt=1)
    logits = torch.randn(
        (T, N, C), dtype=torch.float32, device=device, generator=logits_generator
    )
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).to(cur_dtype)
    targets, target_lengths = _make_targets(
        N,
        S,
        C,
        device,
        blank=blank,
        generator=targets_generator,
    )
    input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
    yield log_probs, targets, input_lengths, target_lengths

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        concat_targets = torch.cat(
            [targets[i, : int(target_lengths[i].item())] for i in range(N)], dim=0
        )
        yield log_probs, concat_targets, input_lengths, target_lengths
        yield log_probs, targets, input_lengths, target_lengths, {
            "reduction": "sum",
            "blank": blank,
        }


CTC_LOSS_BENCH_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_bf16:
    CTC_LOSS_BENCH_DTYPES.append(torch.bfloat16)


def _reference_ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    *args,
    **kwargs,
):
    ref_log_probs = log_probs if log_probs.dtype == torch.float32 else log_probs.float()
    return torch.nn.functional.ctc_loss(
        ref_log_probs,
        targets,
        input_lengths,
        target_lengths,
        *args,
        **kwargs,
    )


class CTCLossBenchmark(base.Benchmark):
    DEFAULT_DTYPES = [torch.float32]
    DEFAULT_SHAPE_DESC = "T, N, C, S"
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        log_probs, targets, input_lengths, target_lengths = args
        total_bytes = sum(
            t.numel() * t.element_size()
            for t in (log_probs, targets, input_lengths, target_lengths)
        )
        return total_bytes * 1e-9 / (latency * 1e-3)

    def get_input_iter(self, cur_dtype) -> Generator:
        for shape in self.shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.ctc_loss
def test_ctc_loss_forward_benchmark():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_LOSS_BENCH_DTYPES,
    )
    bench.run()


@pytest.mark.ctc_loss
def test_ctc_loss_backward_benchmark():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=_reference_ctc_loss,
        dtypes=CTC_LOSS_BENCH_DTYPES,
        is_backward=True,
    )
    bench.run()
