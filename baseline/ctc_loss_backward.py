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

"""NVIDIA native baseline for `ctc_loss_backward`.

Mirrors benchmark/test_ctc_loss.py (CtcLossBenchmark, a GenericBenchmark
subclass):
    op_name    = "ctc_loss" + "_backward" (appended by base.Benchmark)
    torch_op   = _ctc_loss_reference (the module-local reference below)
    dtypes     = [torch.float32, torch.float16]
    is_backward = True
    shapes     = yaml:ctc_loss_backward (benchmark/core_shapes.yaml),
                 shape_desc "T, N, C, S", then filtered by
                 Benchmark.MAX_BACKWARD_ELEMENTS (2**27)

The benchmark's reference upcasts half inputs to float32 for the F.ctc_loss
call and casts the result back, which is what `_ctc_loss_reference` does here.

ctc_loss_input_fn yields two collection points per shape at --level core: one
with padded targets [N, S] and one with the same targets concatenated into a
1-D buffer. The reduction="sum" variant is COMPREHENSIVE-only and is not part
of this grid. Target lengths follow the benchmark's deterministic
`max(1, S - (row % 5))` rule.

Only the autograd backward is timed; config() describes the forward call
because that is what the benchmark records for --base-data matching.
"""

import torch

OP_NAME = "ctc_loss_backward"
DTYPES = [torch.float32, torch.float16]
IS_INPLACE = False
NATIVE_UNAVAILABLE_REASON = (
    "torch.nn.functional.ctc_loss is unavailable on this torch build"
)

# yaml:ctc_loss_backward -- shape_desc "T, N, C, S"
_SHAPES = [
    (64, 4, 32, 16),
    (256, 16, 64, 48),
    (512, 32, 64, 48),
]

_TARGET_LAYOUTS = ("padded", "concatenated")
_BLANK = 0
_REDUCTION = "mean"
_ZERO_INFINITY = False


def _resolve(path):
    """Walk a dotted attribute path from the torch root; None when absent.

    torch.ops.aten.<name>.<overload> raises for an operator this build does not
    carry, so the lookup is guarded.
    """
    obj = torch
    try:
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
    except (AttributeError, RuntimeError):
        return None
    return obj


def _ctc_loss_reference(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
):
    """benchmark/test_ctc_loss.py::_ctc_loss_reference -- half upcast to fp32."""
    original_dtype = log_probs.dtype
    work_log_probs = log_probs
    if work_log_probs.dtype in (torch.float16, torch.bfloat16):
        work_log_probs = work_log_probs.float()
    out = torch.nn.functional.ctc_loss(
        work_log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=blank,
        reduction=reduction,
        zero_infinity=zero_infinity,
    )
    return out.to(original_dtype) if out.dtype != original_dtype else out


def _make_targets(batch, max_target, classes, device, target_layout):
    """benchmark/test_ctc_loss.py::_make_targets -- deterministic targets."""
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


def _backward(out, xs, dout):
    """The timed region for an is_backward benchmark.

    base.Benchmark._benchmark_callable runs the forward once outside the timed
    region, draws dout = randn_like(out), collects the args that require grad
    and then times only torch.autograd.grad(...). build_inputs() does the same
    preparation, so no forward work lands inside the measured call.
    """
    return torch.autograd.grad((out,), xs, grad_outputs=(dout,), retain_graph=True)


def native():
    if _resolve("torch.nn.functional.ctc_loss") is None:
        return None
    return _backward


def grid():
    return [
        {"shape": list(s), "target_layout": layout}
        for s in _SHAPES
        for layout in _TARGET_LAYOUTS
    ]


def build_inputs(binding, dtype, device):
    t_steps, batch, classes, max_target = binding["shape"]
    raw = torch.randn(t_steps, batch, classes, dtype=torch.float32, device=device)
    log_probs = raw.log_softmax(-1).to(dtype).requires_grad_(True)
    input_lengths = torch.full((batch,), t_steps, dtype=torch.long, device=device)
    targets, target_lengths = _make_targets(
        batch, max_target, classes, device, binding["target_layout"]
    )
    out = _ctc_loss_reference(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=_BLANK,
        reduction=_REDUCTION,
        zero_infinity=_ZERO_INFINITY,
    )
    dout = torch.randn_like(out)
    return (out, [log_probs], dout), {}


def key_shape(binding):
    shape = "x".join(str(d) for d in binding["shape"])
    return f"{shape}_{binding['target_layout']}"


def config(binding, dtype):
    t_steps, batch, classes, max_target = binding["shape"]
    layout = binding["target_layout"]
    if layout == "padded":
        target_shape = [batch, max_target]
    else:
        total = sum(max(1, max_target - (row % 5)) for row in range(batch))
        target_shape = [total]
    dt = str(dtype)
    return {
        "inputs": {
            "args": [
                {
                    "shape": [t_steps, batch, classes],
                    "dtype": dt,
                    "requires_grad": True,
                },
                {"shape": target_shape, "dtype": "torch.int64"},
                {"shape": [batch], "dtype": "torch.int64"},
                {"shape": [batch], "dtype": "torch.int64"},
            ],
            "kwargs": {
                "blank": {"scalar": _BLANK},
                "reduction": _REDUCTION,
                "zero_infinity": {"scalar": _ZERO_INFINITY},
            },
        },
        "outputs": {
            "grad_log_probs": {
                "shape": [t_steps, batch, classes],
                "dtype": dt,
            }
        },
        "dims": {
            "T": t_steps,
            "N": batch,
            "C": classes,
            "S": max_target,
            "target_layout": layout,
        },
    }
