from typing import Generator

import pytest
import torch

from benchmark.attri_util import BenchLevel
from benchmark.performance_utils import Config, GenericBenchmark


def ctc_loss_input_fn(shape, cur_dtype, device):
    T, N, C = shape
    S = max(1, T // 3)
    log_probs = torch.randn(T, N, C, dtype=cur_dtype, device=device)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    targets = torch.randint(1, C, (N, S), dtype=torch.long, device=device)
    input_lengths = torch.full((N,), T, dtype=torch.long, device=device)
    target_lengths = torch.full((N,), S, dtype=torch.long, device=device)
    yield log_probs, targets, input_lengths, target_lengths, {
        "blank": 0,
        "reduction": "sum",
    }
    if Config.bench_level == BenchLevel.COMPREHENSIVE:
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": "none"},
        )
        yield (
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            {"blank": 0, "reduction": "mean"},
        )


class CTCLossBenchmark(GenericBenchmark):
    DEFAULT_SHAPE_DESC = "T, N, C"

    def get_input_iter(self, cur_dtype) -> Generator:
        ctc_shapes = [
            (10, 1, 32),
            (20, 4, 64),
            (50, 8, 128),
            (100, 4, 256),
            (100, 16, 128),
            (150, 16, 512),
            (200, 8, 1024),
            (300, 32, 128),
        ]
        for shape in ctc_shapes:
            yield from self.input_fn(shape, cur_dtype, self.device)


@pytest.mark.ctc_loss
def test_perf_ctc_loss():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[torch.float32],
    )
    bench.run()


@pytest.mark.ctc_loss
def test_perf_ctc_loss_backward():
    bench = CTCLossBenchmark(
        input_fn=ctc_loss_input_fn,
        op_name="ctc_loss",
        torch_op=torch.nn.functional.ctc_loss,
        dtypes=[torch.float32],
        is_backward=True,
    )
    bench.run()
