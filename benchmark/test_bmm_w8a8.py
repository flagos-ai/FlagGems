# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import torch

import flag_gems

from . import base
from .test_blas_perf_parallel import ParallelBenchmarkMixin


def make_channel_int8_bmm_inputs(
    shape: tuple[int, int, int, int], device: str, dtype: torch.dtype = torch.int8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, rows, columns, reduction = shape
    torch.manual_seed(0)
    activation = torch.randint(
        -128, 128, (batch, rows, reduction), device=device, dtype=dtype
    )
    weight = torch.randint(
        -128, 128, (batch, columns, reduction), device=device, dtype=dtype
    ).transpose(1, 2)
    activation_scale = torch.rand((batch, rows), device=device) / 127
    weight_scale = torch.rand((batch, columns), device=device) / 127
    return activation, weight, activation_scale, weight_scale


def bf16_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return torch.bmm(baseline_activation, baseline_weight)


def native_int8_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> list[torch.Tensor]:
    import torch_npu

    return [
        torch_npu.npu_quant_matmul(
            activation[index],
            weight[index],
            weight_scale[index],
            pertoken_scale=activation_scale[index],
            output_dtype=torch.bfloat16,
        )
        for index in range(activation.shape[0])
    ]


def channel_int8_bmm(
    activation: torch.Tensor,
    weight: torch.Tensor,
    activation_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    baseline_activation: torch.Tensor,
    baseline_weight: torch.Tensor,
) -> torch.Tensor:
    return flag_gems.bmm_w8a8(activation, weight, activation_scale, weight_scale)


class ChannelInt8BmmBenchmark(ParallelBenchmarkMixin, base.Benchmark):
    """Time BMM on prepared INT8 and BF16 inputs with the shared timer."""

    RECORD_PARALLEL_RESULTS = True
    DEFAULT_SHAPE_DESC = "B, M, N, K"
    DEFAULT_SHAPE_FILES = str(Path(__file__).with_name("core_shapes.yaml"))

    def get_input_iter(self, dtype):
        for shape in self.shapes:
            activation, weight, activation_scale, weight_scale = (
                make_channel_int8_bmm_inputs(shape, self.device, dtype)
            )
            # Both timed APIs receive prepared inputs; conversion stays outside timing.
            baseline_activation = (
                activation.float() * activation_scale[:, :, None]
            ).to(torch.bfloat16)
            baseline_weight = (
                weight.transpose(1, 2).float() * weight_scale[:, :, None]
            ).to(torch.bfloat16)
            yield (
                activation,
                weight,
                activation_scale,
                weight_scale,
                baseline_activation,
                baseline_weight.transpose(1, 2),
            )

    def should_forward_parallel_dtype(self, dtype_name):
        # Workers select their INT8 default without extending the shared CLI choices.
        return False


@pytest.mark.bmm_w8a8
@pytest.mark.parametrize(
    "baseline",
    ["bf16", "native_int8"] if "bmm_w8a8_int8" in vars(flag_gems) else ["bf16"],
)
@pytest.mark.skipif(
    "bmm_w8a8" not in vars(flag_gems),
    reason="backend does not export channel-wise INT8 BMM",
)
def test_perf_bmm_w8a8(baseline):
    torch_op = native_int8_bmm if baseline == "native_int8" else bf16_bmm
    bench = ChannelInt8BmmBenchmark(
        op_name="bmm_w8a8", torch_op=torch_op, dtypes=[torch.int8]
    )
    bench.set_gems(channel_int8_bmm)
    bench.run()
