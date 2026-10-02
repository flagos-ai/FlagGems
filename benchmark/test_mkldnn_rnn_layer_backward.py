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


"""Benchmark for aten::mkldnn_rnn_layer_backward.

The native operator is CPU-registered and consumes the opaque uint8 workspace the
training forward produced, so the builders allocate CPU operands directly and the
forward that creates the workspace runs inside the builder, outside the measured
call.
"""

import math

import pytest
import torch

import flag_gems

from . import base

# (T, N, I, H) descriptors: the input is rank-3 (T, N, I) and the hidden size
# drives every weight, bias, state and gradient shape.
LSTM_BENCH_SHAPES = [
    (2, 19, 7, 8),
    (16, 7, 57, 32),
    (20, 320, 15, 16),
    (16, 128, 64, 60),
]

# Probed: the native backward accepts float32 / bfloat16 / float16 only.
BENCH_DTYPES = [torch.float32, torch.bfloat16, torch.float16]


def _requested_shape(entry):
    # A shape-file entry is an extent sequence or a descriptor dict carrying the
    # same request under "input"/"shape", with "hidden" overriding the derived
    # hidden size.  Tuples keep a caller-supplied list and a shared-grid tuple on
    # one canonical case.
    if isinstance(entry, dict):
        for key in ("input", "shape"):
            if key in entry:
                return tuple(entry[key]), entry.get("hidden")
        raise ValueError("a descriptor needs an input or shape field")
    return tuple(entry), None


def _descriptor(entry):
    """Map a requested case shape onto a legal (T, N, I, H) LSTM geometry.

    The primitive needs a sequence, batch, input and hidden extent of at least
    one, so an empty entry is rejected statically instead of being clamped into a
    different workload.  A rank-3 entry is the sequence operand itself; a rank the
    primitive cannot express preserves every input element by adding leading unit
    axes or folding leading axes into the sequence; unspecified hidden width is one.
    """
    shape, hidden = _requested_shape(entry)
    if any(extent == 0 for extent in shape):
        raise ValueError(f"mkldnn_rnn_layer needs T, N, I, H >= 1, got {shape}")
    if len(shape) == 4:
        t, n, i, h = shape
    elif len(shape) == 3:
        # A bare rank-three row specifies only (T, N, I), not hidden width.
        # Use the same unit hidden width as other bare input shapes; explicit
        # four-field descriptors and dict hidden overrides retain their H.
        t, n, i = shape
        h = 1
    elif len(shape) == 2:
        t, n, i, h = 1, shape[0], shape[1], 1
    elif len(shape) == 1:
        # A flat extent specifies no RNN axes. Factor its element count across
        # batch and features so neither an enormous state batch nor packed
        # feature weights are invented; retain every requested input element.
        i = math.isqrt(shape[0])
        while shape[0] % i:
            i -= 1
        t, n, h = 1, shape[0] // i, 1
    elif not shape:
        t, n, i, h = 1, 1, 1, 1
    else:
        t, n, i, h = math.prod(shape[:-2]), shape[-2], shape[-1], 1
    return (t, n, i, h if hidden is None else hidden)


def _dedupe(shapes):
    # Descriptor dicts are unhashable, so equality is used instead of a set.
    unique = []
    for shape in shapes:
        if shape not in unique:
            unique.append(shape)
    return unique


def _case_fn(shape, dtype):
    del dtype
    t, n, i, h = _descriptor(shape)
    yield base.BenchmarkCasePlan(
        shape={"input": [t, n, i], "hidden": h},
        params={
            "mode": 2,
            "hidden_size": h,
            "num_layers": 1,
            "has_biases": True,
            "train": True,
            "reverse": False,
            "bidirectional": False,
            "batch_first": False,
        },
        builder_args=((t, n, i, h),),
    )


def _build_inputs_fn(plan, dtype, device):
    # device is unused: the native operator consumes CPU operands, so the inputs
    # are allocated on CPU and no transfer lands inside the measured call.
    del device
    params = plan.params
    t, n, i, h = plan.builder_args[0]

    def cpu_tensor(sizes):
        return torch.rand(sizes, dtype=dtype, device="cpu")

    inp = cpu_tensor((t, n, i))
    weight_ih = cpu_tensor((4 * h, i))
    weight_hh = cpu_tensor((4 * h, h))
    bias_ih = cpu_tensor((4 * h,))
    bias_hh = cpu_tensor((4 * h,))
    hx = cpu_tensor((n, h))
    cx = cpu_tensor((n, h))

    # GradMode decides whether the forward allocates the workspace, which cannot
    # be synthesized from the other arguments.
    with torch.enable_grad():
        output, hy, cy, workspace = torch.ops.aten.mkldnn_rnn_layer(
            inp,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            hx,
            cx,
            params["reverse"],
            [],
            params["mode"],
            h,
            params["num_layers"],
            params["has_biases"],
            params["bidirectional"],
            params["batch_first"],
            params["train"],
        )

    args = [
        inp,
        weight_ih,
        weight_hh,
        bias_ih,
        bias_hh,
        hx,
        cx,
        output,
        hy,
        cy,
        cpu_tensor(tuple(output.shape)),
        cpu_tensor(tuple(hy.shape)),
        cpu_tensor(tuple(cy.shape)),
        params["reverse"],
        params["mode"],
        h,
        params["num_layers"],
        params["has_biases"],
        params["train"],
        params["bidirectional"],
        [],
        params["batch_first"],
        workspace,
    ]
    # unpack_to_args_kwargs takes flat positional arguments plus a trailing kwargs
    # dict, not an (args, kwargs) pair.
    return (*args, {})


class MkldnnRnnLayerBackwardBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # The shared grid and any caller-supplied shape file stay in place; the
        # native-valid LSTM descriptors are unioned on top of them and every entry
        # is normalized to a tuple so a caller list and a shared tuple address one
        # canonical case.
        super().set_shapes(shape_file_path)
        self.shapes = _dedupe(
            [
                shape if isinstance(shape, dict) else tuple(shape)
                for shape in self.shapes
            ]
            + LSTM_BENCH_SHAPES
        )


@pytest.mark.mkldnn_rnn_layer_backward
def test_mkldnn_rnn_layer_backward():
    bench = MkldnnRnnLayerBackwardBenchmark(
        op_name="mkldnn_rnn_layer_backward",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.mkldnn_rnn_layer_backward,
        gems_op=getattr(flag_gems, "mkldnn_rnn_layer_backward", None),
        dtypes=BENCH_DTYPES,
    )
    bench.run()
