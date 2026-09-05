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

from . import base

# The two registered ATen variants are ``aten::_fused_adagrad`` (out-of-place)
# and ``aten::_fused_adagrad_`` (in-place).  pytest forbids marker names that
# start with ``_``, so the leading-underscore ATen names cannot be used as
# marks; the stripped, pytest-legal marks (``fused_adagrad`` /
# ``fused_adagrad_``) are applied to the benchmark functions below, matching
# the ``fused_adam`` convention.

# Shapes representing realistic optimizer parameter sizes:
# small embedding / attention weights / MLP weights / large embedding.
_FUSED_ADAGRAD_SHAPES = [
    (256, 256),
    (512, 512),
    (1024, 256),
    (2048, 512),
    (4096, 256),
    (65536,),
]

# Common optimizer hyper-parameters used by both the torch native and the
# FlagGems implementations so the comparison is like-for-like.
_LR = 0.01
_LR_DECAY = 0.0
_WEIGHT_DECAY = 0.0
_EPS = 1e-10
_MAXIMIZE = False


class FusedAdagradBenchmark(base.GenericBenchmark):
    # fused_adagrad uses 4 tensors per case (param, grad, state_sum, step)
    # so shapes are kept moderate to avoid OOM on CI GPUs.
    DEFAULT_SHAPES = _FUSED_ADAGRAD_SHAPES

    def set_shapes(self, shape_file=None):
        self.shapes = list(_FUSED_ADAGRAD_SHAPES)


def fused_adagrad_input_fn(shape, dtype, device):
    param = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    state_sum = torch.zeros(shape, dtype=dtype, device=device)
    state_step = torch.tensor([3.0], dtype=torch.float32, device=device)
    yield param, grad, state_sum, state_step


def torch_op(param, grad, state_sum, state_step):
    """Native torch fused Adagrad step (in-place)."""
    torch._fused_adagrad_(
        [param],
        [grad],
        [state_sum],
        [state_step],
        lr=_LR,
        lr_decay=_LR_DECAY,
        weight_decay=_WEIGHT_DECAY,
        eps=_EPS,
        maximize=_MAXIMIZE,
    )
    return param


@pytest.mark.fused_adagrad_
def test_fused_adagrad_():
    def gems_op(param, grad, state_sum, state_step):
        flag_gems._fused_adagrad_(
            [param],
            [grad],
            [state_sum],
            [state_step],
            lr=_LR,
            lr_decay=_LR_DECAY,
            weight_decay=_WEIGHT_DECAY,
            eps=_EPS,
            maximize=_MAXIMIZE,
        )
        return param

    bench = FusedAdagradBenchmark(
        input_fn=fused_adagrad_input_fn,
        op_name="fused_adagrad_",
        torch_op=torch_op,
        # _fused_adagrad only supports float32 for optimizer state precision
        dtypes=[torch.float32],
    )
    bench.set_gems(gems_op)
    bench.run()
