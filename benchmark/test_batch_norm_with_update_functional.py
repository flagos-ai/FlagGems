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
from _pytest.mark.structures import Mark, MarkDecorator

from . import base, consts

# ``_batch_norm_with_update_functional`` starts with an underscore, and ``pytest.mark``
# refuses to generate a marker via attribute access for such names. Register it
# directly on the MarkGenerator so ``@pytest.mark._batch_norm_with_update_functional`` and
# ``-m _batch_norm_with_update_functional`` both work.
setattr(
    pytest.mark,
    "_batch_norm_with_update_functional",
    MarkDecorator(
        Mark("_batch_norm_with_update_functional", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)


class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            # 3D shapes represented as [batch_size, channels, hidden_size]
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            # 4D shapes represented as [batch_size, channels, H, W]
            (1, 8, 4, 4),
            (16, 8, 128, 128),
        ]


@pytest.mark.batch_norm_with_update_functional
@pytest.mark.underscore_batch_norm_with_update_functional
def test__batch_norm_with_update_functional():
    def case_fn(shape, dtype):
        yield base.BenchmarkCasePlan(
            shape={"input": list(shape)},
            params={"momentum": 0.1, "eps": 1e-5},
            builder_args=(shape,),
        )

    def build_inputs_fn(plan, dtype, device):
        (shape,) = plan.builder_args
        C = shape[1]
        inp = torch.randn(shape, dtype=dtype, device=device)
        weight = torch.randn((C,), dtype=dtype, device=device)
        bias = torch.randn((C,), dtype=dtype, device=device)
        running_mean = torch.randn((C,), dtype=dtype, device=device)
        running_var = torch.abs(torch.randn((C,), dtype=dtype, device=device)) + 0.1
        momentum = 0.1
        eps = 1e-5
        return inp, weight, bias, running_mean, running_var, momentum, eps

    bench = NormBenchmark(
        case_fn=case_fn,
        build_inputs_fn=build_inputs_fn,
        op_name="_batch_norm_with_update_functional",
        torch_op=torch.ops.aten._batch_norm_with_update_functional,
        dtypes=consts.FLOAT_DTYPES,
    )
    from flag_gems.ops._batch_norm_with_update_functional import (
        _batch_norm_with_update_functional as gems_bn,
    )

    bench.set_gems(gems_bn)
    bench.run()
