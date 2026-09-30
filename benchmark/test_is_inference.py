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

from . import base, consts, utils

# aten::is_inference(Tensor self) -> bool reads one tensor's inference flag, so
# the only workload dimension besides the shape is the state of that tensor.
# Every shared shape is planned twice (plain and torch.inference_mode()); the
# plans carry only JSON metadata, so --list-cases stays tensor-free.


def _case_fn(shape, dtype):
    del dtype
    for inference in (False, True):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"inference": inference},
            builder_args=(shape, inference),
        )


def _build_inputs_fn(plan, dtype, device):
    shape, inference = plan.builder_args
    inp = utils.generate_tensor_input(shape, dtype, device)
    if inference:
        # A clone created inside inference mode is an inference tensor, which is
        # the state this plan measures.
        with torch.inference_mode():
            inp = inp.clone()
    return inp, {}


@pytest.mark.is_inference
def test_is_inference():
    bench = base.GenericBenchmark(
        op_name="is_inference",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.is_inference,
        gems_op=getattr(flag_gems, "is_inference", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
