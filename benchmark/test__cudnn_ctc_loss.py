# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Benchmark for torch.ops.aten._cudnn_ctc_loss.

Only float32 is supported natively (log_probs must be a compute-device float32
tensor and targets a CPU int32 one), so consts.FLOAT_DTYPES is not usable here
and the dtype list is fixed to float32. Timed and listed cases come from one
case_fn / build_inputs_fn pair, so --list-cases allocates no tensors and
--case-id replay uses exactly the plans a normal run uses.
"""

import pytest
import torch
from _pytest.mark.structures import Mark, MarkDecorator

import flag_gems

from . import base, utils
from .generated_operator_utils import OperatorBenchmark

setattr(
    pytest.mark,
    "_cudnn_ctc_loss",
    MarkDecorator(Mark("_cudnn_ctc_loss", (), {}, _ispytest=True), _ispytest=True),
)

# _cudnn_ctc_loss has no entry in core_shapes.yaml, so the ctc_loss rows from
# that file (shape_desc 'T, N, C, S') are supplied as defaults; an explicit
# --shape-file still takes precedence through set_shapes(shape_file_path).
_CUDNN_CTC_LOSS_SHAPES = [
    (64, 4, 32, 16),
    (256, 16, 64, 48),
    (512, 32, 64, 48),
    (1024, 32, 128, 96),
]

# Every descriptor is run in three variants: a cycled transcript, a repeated
# single-label transcript (a length-S target can need more than S frames), and
# cuDNN's deterministic algorithm selection on the cycled transcript.
_CASE_VARIANTS = (
    ("cycle", False),
    ("repeat", False),
    ("cycle", True),
)


def _validate_descriptor(descriptor):
    """Validate and return a (T, N, C, S) ctc-loss descriptor.

    bool is an int subclass, so it is rejected explicitly. C == 1 is a native
    BAD_PARAM and would divide by zero in the target builder. S > T is allowed:
    native accepts it, and repeated labels can need more than S frames.
    """
    if not isinstance(descriptor, (tuple, list)) or len(descriptor) != 4:
        raise ValueError(f"ctc-loss descriptor must be (T, N, C, S): {descriptor!r}")
    for name, value in zip(("T", "N", "C", "S"), descriptor):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"descriptor {name} must be an int, got {value!r}")
    time_steps, batch, num_classes, target_length = descriptor
    if time_steps < 1:
        raise ValueError(f"T must be >= 1, got {time_steps}")
    if batch < 1:
        raise ValueError(f"N must be >= 1, got {batch}")
    if num_classes < 2:
        raise ValueError(
            f"C must be >= 2 (C == 1 is a native BAD_PARAM), got {num_classes}"
        )
    if target_length < 0:
        raise ValueError(f"S must be >= 0, got {target_length}")
    return (time_steps, batch, num_classes, target_length)


class _CudnnCtcLossBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path,
            default_shapes=[
                _validate_descriptor(descriptor)
                for descriptor in _CUDNN_CTC_LOSS_SHAPES
            ],
        )


def _case_fn(shape, dtype):
    del dtype
    time_steps, batch, num_classes, target_length = _validate_descriptor(shape)
    for pattern, deterministic in _CASE_VARIANTS:
        yield base.BenchmarkCasePlan(
            shape={"input": (time_steps, batch, num_classes)},
            params={
                "target_length": target_length,
                "pattern": pattern,
                "deterministic": deterministic,
                "blank": 0,
                "zero_infinity": True,
            },
            builder_args=(
                time_steps,
                batch,
                num_classes,
                target_length,
                pattern,
                deterministic,
            ),
        )


def _build_inputs_fn(plan, dtype, device):
    (
        time_steps,
        batch,
        num_classes,
        target_length,
        pattern,
        deterministic,
    ) = plan.builder_args
    log_probs = utils.generate_tensor_input(
        (time_steps, batch, num_classes), dtype, device
    )
    # native targets are a 1-D CPU int32 label tensor below C whose length is
    # sum(target_lengths); an empty one is valid for the all-empty transcript
    total = target_length * batch
    if pattern == "cycle":
        targets = torch.arange(total, dtype=torch.int32) % (num_classes - 1) + 1
    else:
        targets = torch.ones(total, dtype=torch.int32)
    return (
        log_probs,
        targets,
        {
            "input_lengths": [time_steps] * batch,
            "target_lengths": [target_length] * batch,
            "blank": plan.params["blank"],
            "deterministic": plan.params["deterministic"],
            "zero_infinity": plan.params["zero_infinity"],
        },
    )


@pytest.mark._cudnn_ctc_loss
def test__cudnn_ctc_loss():
    bench = _CudnnCtcLossBenchmark(
        op_name="_cudnn_ctc_loss",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cudnn_ctc_loss,
        gems_op=getattr(flag_gems, "_cudnn_ctc_loss", None),
        dtypes=[torch.float32],
    )
    bench.run()
