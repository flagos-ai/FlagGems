# Copyright 2026, The FlagGems Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Benchmark for ``torch.ops.aten._cudnn_rnn_flatten_weight``.

One case is one RNN weight-list geometry. The native operator is cuDNN only, so
``torch_op`` needs a CUDA reference device; ``fresh_inputs`` is required because
the operator rewrites the list entries in place, and a cached already-flattened
list would time a no-op.
"""

import pytest
import torch

import flag_gems

from . import base
from .generated_operator_utils import OperatorBenchmark

# cuDNN mode -> number of gates per hidden unit.
_GATES = {0: 1, 1: 1, 2: 4, 3: 3}

# config = (mode, input_size, hidden_size, num_layers, proj_size,
#           bidirectional, bias, batch_first)
DEFAULT_CONFIGS = [
    (2, 128, 256, 1, 0, False, True, False),
    (2, 512, 512, 2, 0, False, True, False),
    (2, 256, 256, 2, 0, True, True, False),
    (2, 512, 256, 1, 64, False, True, False),
    (3, 256, 256, 2, 0, True, True, False),
    (1, 256, 512, 3, 0, False, True, False),
    # ``weight_stride0 == 2`` biasless list and a ``batch_first`` variant.
    (2, 512, 512, 1, 0, False, False, False),
    (3, 256, 256, 2, 0, True, True, True),
]

_DTYPES = [torch.float16, torch.float32]
if flag_gems.runtime.device.support_fp64:
    _DTYPES.append(torch.float64)


def _normalize_config(config):
    """Validate one descriptor before any tensor is allocated.

    Every integral field, ``mode`` included, rejects ``bool`` and non-integral
    values, so a configuration cannot slip through by comparing equal to a
    supported integer.
    """
    if not isinstance(config, (tuple, list)) or len(config) != 8:
        raise ValueError(f"config must be a sequence of 8 fields, got {config!r}")
    (
        mode,
        input_size,
        hidden_size,
        num_layers,
        proj_size,
        bidirectional,
        bias,
        batch_first,
    ) = config
    for name, value in (
        ("input_size", input_size),
        ("hidden_size", hidden_size),
        ("num_layers", num_layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    # mode 0 (RNN_RELU) is valid, so this checks the integral type only; the
    # supported values are checked against ``_GATES`` below.
    if isinstance(mode, bool) or not isinstance(mode, int):
        raise ValueError(f"mode must be an int, got {mode!r}")
    if isinstance(proj_size, bool) or not isinstance(proj_size, int) or proj_size < 0:
        raise ValueError(f"proj_size must be a non-negative int, got {proj_size!r}")
    if mode not in _GATES:
        raise ValueError(f"unsupported RNN mode {mode!r}")
    if proj_size >= hidden_size:
        raise ValueError(
            f"proj_size {proj_size} must be smaller than hidden_size {hidden_size}"
        )
    if proj_size > 0 and mode != 2:
        raise ValueError("projection is only supported by LSTM (mode 2)")
    for name, value in (
        ("bidirectional", bidirectional),
        ("bias", bias),
        ("batch_first", batch_first),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{name} must be a bool, got {value!r}")
    return (
        mode,
        input_size,
        hidden_size,
        num_layers,
        proj_size,
        bidirectional,
        bias,
        batch_first,
    )


def _weight_stride0(config):
    """List stride: 4/2 without projection, 5/3 with one."""
    _, _, _, _, proj_size, _, bias, _ = config
    if bias:
        return 5 if proj_size > 0 else 4
    return 3 if proj_size > 0 else 2


def _component_shapes(config, weight_stride0):
    """Weight shapes of one list, in cuDNN's (layer, direction) order."""
    mode, input_size, hidden_size, num_layers, proj_size, bidirectional, _, _ = config
    gates = _GATES[mode]
    num_directions = 2 if bidirectional else 1
    real_hidden = proj_size if proj_size > 0 else hidden_size
    has_bias = weight_stride0 >= 4
    has_projection = weight_stride0 in (3, 5)
    shapes = []
    for layer in range(num_layers):
        layer_input = input_size if layer == 0 else real_hidden * num_directions
        for _ in range(num_directions):
            shapes.append((gates * hidden_size, layer_input))
            shapes.append((gates * hidden_size, real_hidden))
            if has_bias:
                shapes.append((gates * hidden_size,))
                shapes.append((gates * hidden_size,))
            if has_projection:
                shapes.append((proj_size, hidden_size))
    return shapes


def _numel(shapes):
    return sum(shape[0] * (shape[1] if len(shape) > 1 else 1) for shape in shapes)


def _case_fn(shape, dtype):
    del dtype
    config = _normalize_config(shape)
    weight_stride0 = _weight_stride0(config)
    shapes = _component_shapes(config, weight_stride0)
    yield base.BenchmarkCasePlan(
        shape={
            "config": str(config),
            "weight_shapes": str(shapes),
            "num_tensors": len(shapes),
            "weight_numel": _numel(shapes),
        },
        params={
            "weight_stride0": weight_stride0,
            "batch_first": config[7],
            "bias": config[6],
        },
        builder_args=(config,),
    )


def _build_inputs_fn(plan, dtype, device):
    config = plan.builder_args[0]
    weight_stride0 = plan.params["weight_stride0"]
    (
        mode,
        input_size,
        hidden_size,
        num_layers,
        proj_size,
        bidirectional,
        _,
        batch_first,
    ) = config
    weights = [
        torch.randn(shape, dtype=dtype, device=device)
        for shape in _component_shapes(config, weight_stride0)
    ]
    # The harness splits this return value element by element: a dict element
    # becomes keyword arguments and every other element becomes ONE positional
    # argument. The ``Tensor[]`` operand must therefore be a direct element, so
    # that it reaches the operator as a Python list instead of a nested tuple.
    return (
        weights,
        weight_stride0,
        input_size,
        mode,
        hidden_size,
        proj_size,
        num_layers,
        batch_first,
        bidirectional,
    )


class CudnnRnnFlattenWeightBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Without a shape file the benchmark's own configurations are used; the
        # public helper already handles the shape-file plus default merge.
        super().set_shapes(shape_file_path, default_shapes=DEFAULT_CONFIGS)


@pytest.mark.cudnn_rnn_flatten_weight
def test__cudnn_rnn_flatten_weight():
    bench = CudnnRnnFlattenWeightBenchmark(
        op_name="_cudnn_rnn_flatten_weight",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cudnn_rnn_flatten_weight,
        gems_op=getattr(flag_gems, "_cudnn_rnn_flatten_weight", None),
        dtypes=_DTYPES,
        fresh_inputs=True,
    )
    bench.run()
