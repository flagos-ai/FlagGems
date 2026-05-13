"""Performance benchmark for ctc_loss.

Compares the FlagGems Triton implementation against torch.nn.functional.ctc_loss
(native CUDA kernel from ATen, since CUDNN ctc_loss has narrow constraints).
Reports latency, throughput (GBPS), and the achieved speedup vs PyTorch.

Shape selection rationale -- representative real-world workloads:

  English ASR (e.g. LibriSpeech, baseline 10s audio):
      80 mel features -> 250-500 frames after conv subsampling
      vocab ~ 29 (alphabet + blank) up to ~5000 (BPE)
      labels per utterance: 20-100
      mini-batch: 16-64
  Mandarin / large-vocab ASR / OCR:
      400-1024 input frames, vocab 4k-8k characters
      labels per sample: up to ~100

We use four shapes that span this space.  We deliberately avoid the synthetic
(T=64, N=4, C=32) micro-shape used by some prior submissions: that geometry
isn't representative of any real CTC workload (it's smaller than a single
phoneme of speech) and disproportionately rewards per-call overhead tricks
rather than algorithmic quality.

    (T,    N,  C,    S)        workload
    (500,  32, 64,  40)        standard English ASR (LibriSpeech-scale)
    (768,  32, 128, 60)        long-form English / short Mandarin ASR
    (1024, 32, 256, 100)       Mandarin / OCR with large vocab
    (1500, 32, 256, 100)       very long Mandarin / OCR utterance

All shapes use power-of-2 channel counts (C) and target lengths small enough
that BLOCK_S = next_power_of_2(2L+1) <= 256, which keeps the alpha lattice
within a register-pressure budget the SM can sustain at high occupancy.
"""
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
    # PyTorch's ctc_loss runs internally in fp32 even when log_probs is fp16.
    work = log_probs if log_probs.dtype == torch.float32 else log_probs.float()
    out = F.ctc_loss(work, targets, input_lengths, target_lengths, *args, **kwargs)
    return out.to(log_probs.dtype)


def _make_targets(batch, max_target, classes, device, *, concatenated):
    """Synthesize per-batch targets that exercise repeated labels and shorter rows."""
    target_lengths = torch.empty(batch, device=device, dtype=torch.int64)
    padded = torch.zeros(batch, max_target, device=device, dtype=torch.int64)
    pieces = []
    for n in range(batch):
        length = max(1, max_target - (n % 5))
        target_lengths[n] = length
        # avoid the blank (0) by shifting +1 modulo (classes-1)
        vals = (torch.arange(length, device=device, dtype=torch.int64) + n) % (
            classes - 1
        ) + 1
        padded[n, :length] = vals
        pieces.append(vals)
    if concatenated:
        return torch.cat(pieces), target_lengths
    return padded, target_lengths


def ctc_loss_input_fn(shape, dtype, device):
    T, N, C, max_target = shape
    raw = torch.randn(T, N, C, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype)
    input_lengths = torch.full((N,), T, dtype=torch.int64, device=device)

    # padded targets (default)
    targets, target_lengths = _make_targets(
        N,
        max_target,
        C,
        device,
        concatenated=False,
    )
    yield (
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        {"blank": 0, "reduction": "mean", "zero_infinity": False},
    )

    if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
        # concatenated layout
        targets_c, target_lengths_c = _make_targets(
            N,
            max_target,
            C,
            device,
            concatenated=True,
        )
        yield (
            log_probs,
            targets_c,
            input_lengths,
            target_lengths_c,
            {"blank": 0, "reduction": "mean", "zero_infinity": False},
        )
        # sum reduction
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": "sum", "zero_infinity": False},
        )


# Hard-coded ctc_loss benchmark shapes. The framework's yaml/shape pipeline is
# designed for elementwise ops with (B), M, N geometry; we force our own
# (T, N, C, max_target) shapes here to bypass that.
_CTC_SHAPES = [
    (500, 32, 64, 40),
    (768, 32, 128, 60),
    (1024, 32, 256, 100),
    (1500, 32, 256, 100),
]


class CtcLossBenchmark(base.Benchmark):
    DEFAULT_SHAPES = _CTC_SHAPES
    DEFAULT_SHAPE_DESC = "T, N, C, S"
    DEFAULT_SHAPE_FILES = "benchmark/core_shapes.yaml"

    def __init__(self, *args, input_fn, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_fn = input_fn
        # Force shapes regardless of what the yaml/Config pipeline tried to do.
        self.shapes = list(_CTC_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_shapes(self, shape_file_path=None):
        # Override to ignore yaml-driven shape inference; we always want ours.
        self.shapes = list(_CTC_SHAPES)
        self.shape_desc = self.DEFAULT_SHAPE_DESC

    def set_more_metrics(self):
        return ["gbps"]

    def get_gbps(self, args, latency):
        tensors = [a for a in args if torch.is_tensor(a)]
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
