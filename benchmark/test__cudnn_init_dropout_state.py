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
from .generated_operator_utils import OperatorBenchmark

# aten::_cudnn_init_dropout_state has no tensor shape: its output extent comes
# from cudnnDropoutGetStatesSize for the active device, and its only accepted
# dtype is torch.uint8. The collected cases therefore vary the operator's real
# parameters and carry descriptive shape metadata.
_DROPOUT_CASES = (
    (0.05, 1),
    (0.1, 7),
    (0.25, 42),
    (0.5, 1234),
    (0.75, 2**31 - 1),
    (0.9, 12345),
)

_STATE_SHAPE = {"state": "cudnn_dropout_rng_state"}


def _case_fn(descriptor, dtype):
    del dtype
    dropout, train, seed = descriptor
    yield base.BenchmarkCasePlan(
        shape=dict(_STATE_SHAPE),
        params={"dropout": dropout, "train": train, "seed": seed},
        builder_args=(dropout, train, seed),
    )


def _build_inputs_fn(plan, dtype, device):
    # Factory form: flat positional arguments plus a trailing kwargs dict, which
    # is the tuple layout Benchmark.unpack_to_args_kwargs expects.
    dropout, train, seed = plan.builder_args
    return dropout, train, seed, {"dtype": dtype, "device": device}


def _build_inputs_fn_out(plan, dtype, device):
    # The .out runtime schema is
    # (float dropout, bool train, int dropout_seed, *, Tensor(a!) out), so the
    # buffer carries dtype/device and no factory option may be passed.
    dropout, train, seed = plan.builder_args
    out = torch.empty(0, dtype=dtype, device=device)
    return dropout, train, seed, {"out": out}


class CudnnInitDropoutStateBenchmark(OperatorBenchmark):
    """Two-phase benchmark for a factory that has no tensor input."""

    def set_shapes(self, shape_file_path=None):
        # The real scalar parameters define each case; cuDNN determines output size.
        super().set_shapes(
            shape_file_path,
            default_shapes=[(dropout, True, seed) for dropout, seed in _DROPOUT_CASES],
        )

    def set_more_shapes(self):
        # The cases ignore the shape, so comprehensive mode adds no shape set.
        return []


# flag_gems may not export this operator yet; getattr(..., None) keeps listing
# working, while execution still requires the KernelGen override to supply a
# real candidate.
@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state():
    bench = CudnnInitDropoutStateBenchmark(
        op_name="_cudnn_init_dropout_state",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._cudnn_init_dropout_state,
        gems_op=getattr(flag_gems, "_cudnn_init_dropout_state", None),
        dtypes=[torch.uint8],
    )
    bench.run()


@pytest.mark.cudnn_init_dropout_state
def test__cudnn_init_dropout_state_out():
    bench = CudnnInitDropoutStateBenchmark(
        op_name="_cudnn_init_dropout_state",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn_out,
        torch_op=torch.ops.aten._cudnn_init_dropout_state.out,
        gems_op=getattr(flag_gems, "_cudnn_init_dropout_state", None),
        dtypes=[torch.uint8],
    )
    bench.run()
