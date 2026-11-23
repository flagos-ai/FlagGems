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

import pytest
import torch

import flag_gems

from . import base, consts

# The registered ATen variant exercised here is ``aten::_fused_sgd_`` (in-place).
# pytest forbids marker names that start with ``_``, so the leading-underscore
# ATen name cannot be used as a mark; the stripped, pytest-legal mark
# (``fused_sgd_``) is applied to the benchmark functions below, matching the
# ``fused_adam`` convention.

# Fused-SGD operates over a *list* of parameter tensors. Each shape is
# multiplied by NUM_PARAMS * 3 (params + grads + momentum buffers), so we keep
# the shapes modest to bound GPU memory. A dedicated subclass pins the shape
# list and disables the very large comprehensive shapes that the base
# ``GenericBenchmark`` would otherwise append.
NUM_PARAMS = 4

# Optimizer-step latency is dominated by the parameter count, not spatial
# layout; modest 2-D and 3-D sizes keep the benchmark fast while exercising
# the fp16/bf16/fp32 compute paths.
SGD_SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
]


class FusedSgdBenchmark(base.GenericBenchmark):
    """GenericBenchmark restricted to modest SGD-friendly shapes."""

    def set_more_shapes(self):
        # The fused-SGD kernel works on a list of tensors; the base class'
        # comprehensive shapes (e.g. (2**28,)) blow up GPU memory once multiplied
        # by NUM_PARAMS * 3, so we do not add any extra shapes here.
        return []

    def set_shapes(self, shape_file_path=None):
        # Override the base shape selection so the fused-SGD benchmark always
        # uses the modest, optimizer-sized shapes defined below instead of the
        # very large pointwise DEFAULT_SHAPES (which include a 1B-element
        # tensor that is far too big for a multi-tensor optimizer benchmark).
        self.shapes = list(SGD_SHAPES)


def _input_fn(shape, dtype, device):
    params = [torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)]
    grads = [torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)]
    momentum_bufs = [
        torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)
    ]
    kwargs = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    yield params, grads, momentum_bufs, kwargs


def _input_fn_tensor_lr(shape, dtype, device):
    params = [torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)]
    grads = [torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)]
    momentum_bufs = [
        torch.randn(shape, dtype=dtype, device=device) for _ in range(NUM_PARAMS)
    ]
    lr = torch.tensor(0.1, dtype=torch.float32, device=device)
    kwargs = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=lr,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    yield params, grads, momentum_bufs, kwargs


def _torch_op(params, grads, momentum_bufs, **kwargs):
    torch._fused_sgd_(params, grads, momentum_bufs, **kwargs)


def _torch_op_tensor_lr(params, grads, momentum_bufs, **kwargs):
    # The tensor-lr overload is reached via the aten namespace.
    torch.ops.aten._fused_sgd_.tensor_lr(params, grads, momentum_bufs, **kwargs)


def _gems_op(params, grads, momentum_bufs, **kwargs):
    flag_gems._fused_sgd_(params, grads, momentum_bufs, **kwargs)


def _gems_op_tensor_lr(params, grads, momentum_bufs, **kwargs):
    flag_gems._fused_sgd__tensor_lr(params, grads, momentum_bufs, **kwargs)


@pytest.mark.fused_sgd_
def test_fused_sgd_():
    bench = FusedSgdBenchmark(
        input_fn=_input_fn,
        op_name="fused_sgd_",
        torch_op=_torch_op,
        # Passed explicitly: without it the harness dispatches through torch for
        # both timings, so the "speedup" would compare native against itself.
        gems_op=_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = list(SGD_SHAPES)
    bench.run()


@pytest.mark.fused_sgd_
def test_fused_sgd__tensor_lr():
    bench = FusedSgdBenchmark(
        input_fn=_input_fn_tensor_lr,
        op_name="fused_sgd__tensor_lr",
        torch_op=_torch_op_tensor_lr,
        gems_op=_gems_op_tensor_lr,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.shapes = list(SGD_SHAPES)
    bench.run()


# A realistic optimizer step sees a *heterogeneous* parameter list -- every
# weight in a model at its own size, not NUM_PARAMS copies of one shape. The
# shapes below are one small transformer block's parameter set (embedding table,
# attention projections at mixed widths, MLP layers and biases), which is
# exactly the workload where per-parameter launches differ most from a single
# batched launch: 10 launches vs 1 per step.
HETEROGENEOUS_PARAMS = [
    (512, 512),  # embedding table
    (2048, 512),  # attn qkv weight
    (2048,),  # attn qkv bias
    (512, 2048),  # attn proj weight
    (512,),  # attn proj bias
    (4096, 512),  # mlp up weight
    (512, 4096),  # mlp down weight
    (512,),  # mlp bias
    (1000,),  # 1-D head
    (7, 7, 7),  # small odd tail
]

# How many times the parameter set above is repeated, one entry per case. These
# are the values each case's shape tuple is built from; a real model has
# hundreds of parameters, so the larger cases are the representative ones.
HETERO_REPEATS = [10, 60, 120]


class FusedSgdHeteroBenchmark(FusedSgdBenchmark):
    """FusedSgdBenchmark whose cases are heterogeneous parameter lists.

    The harness iterates over ``self.shapes`` and passes each entry to the input
    function; here each entry is ``(repeats,)``, so ``set_shapes`` has to
    override the base's uniform-shape list rather than the input function
    reinterpreting one of those uniforms.
    """

    def set_shapes(self, shape_file_path=None):
        self.shapes = [(n,) for n in HETERO_REPEATS]


def _hetero_input_fn(shape, dtype, device):
    """Build a mixed-size parameter list repeated ``shape[0]`` times."""
    repeats = shape[0]
    shapes = [
        HETEROGENEOUS_PARAMS[i % len(HETEROGENEOUS_PARAMS)] for i in range(repeats)
    ]
    params = [torch.randn(s, dtype=dtype, device=device) for s in shapes]
    grads = [torch.randn_like(p) for p in params]
    momentum_bufs = [torch.randn_like(p) for p in params]
    kwargs = dict(
        weight_decay=0.01,
        momentum=0.9,
        lr=0.1,
        dampening=0.0,
        nesterov=False,
        maximize=False,
        is_first_step=False,
    )
    yield params, grads, momentum_bufs, kwargs


@pytest.mark.fused_sgd_
def test_fused_sgd__heterogeneous():
    """Batched kernel vs native on realistic mixed-size parameter lists.

    The uniform-shape cases in ``test_fused_sgd_`` cannot show the batching win:
    every parameter has the same size there, so a per-parameter launch and one
    batched launch issue the same total work. This benchmark uses the mixed
    sizes and the parameter counts a real optimizer step has.
    """
    bench = FusedSgdHeteroBenchmark(
        input_fn=_hetero_input_fn,
        op_name="fused_sgd__heterogeneous",
        torch_op=_torch_op,
        gems_op=_gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
