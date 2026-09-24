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

import flag_gems

from . import base, consts, utils
from .generated_operator_utils import OperatorBenchmark

# The operator name starts with an underscore, so pytest cannot resolve the
# marker by attribute lookup; register it on the MarkGenerator.
setattr(
    pytest.mark,
    "_sparse_csr_prod",
    MarkDecorator(Mark("_sparse_csr_prod", (), {}, _ispytest=True), _ispytest=True),
)

# The CSR kernel asserts input_dim == 2 and core_shapes.yaml has no
# _sparse_csr_prod entry, so these 2-D defaults replace the inherited 1-D/3-D
# ones. Inputs come from a dense tensor converted to CSR, so the stored entry
# count depends on how many random values are nonzero. A caller-supplied
# --shape_file entry for the operator still overrides them.
_PROD_SHAPES = [
    (1024, 1024),
    (2048, 2048),
    (4096, 512),
    (512, 4096),
    (1, 65536),
    (65536, 1),
]

# Single-axis and multi-axis (empty dim list) reduction forms share one plan
# shape, so listing and replay cover both without allocating tensors.
_DIMS = ([0], [1], [])

_BENCH_DTYPES = [
    dtype
    for dtype in consts.FLOAT_DTYPES
    if dtype != torch.bfloat16 or flag_gems.runtime.device.support_bf16
]


def _case_fn(shape, dtype):
    del dtype
    for dim in _DIMS:
        yield base.BenchmarkCasePlan(
            shape={"input": shape},
            params={"dim": dim},
            builder_args=(shape,),
        )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device).to_sparse_csr()
    # unpack_to_args_kwargs turns the returned dict into call kwargs, so the
    # reference and the candidate share op(input, dim=..., keepdim=True).
    return inp, {"dim": plan.params["dim"], "keepdim": True}


class SparseCsrProdBenchmark(OperatorBenchmark):
    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_PROD_SHAPES)


@pytest.mark._sparse_csr_prod
def test__sparse_csr_prod():
    bench = SparseCsrProdBenchmark(
        op_name="_sparse_csr_prod",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._sparse_csr_prod.dim_dtype,
        gems_op=getattr(flag_gems, "_sparse_csr_prod", None),
        dtypes=_BENCH_DTYPES,
    )
    bench.run()
