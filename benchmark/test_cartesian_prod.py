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

"""Benchmark for ``aten::cartesian_prod(Tensor[] tensors) -> Tensor``.

``cartesian_prod`` consumes a list of 1-D tensors and writes
``prod(sizes) x len(sizes)`` output elements. The default shape file and
``consts.DEFAULT_SHAPES`` describe dense multi-dim tensors, which is both
meaningless for the 1-D input contract and large enough to exhaust memory, so
this benchmark enumerates its own list-of-input-sizes cases. Each entry is the
list of 1-D input sizes; the largest case writes ~16.8M output elements
(67 MiB for float32).

The two-phase ``case_fn`` / ``build_inputs_fn`` API is used because one case is
a *list* of tensors rather than a single dense shape.
"""

import pytest
import torch

import flag_gems

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

_CARTESIAN_PROD_BENCH_SIZES = (
    (4096,),  # single input -> (4096,)
    (1024, 1024),  # two inputs -> (1048576, 2)
    (64, 64, 64),  # three inputs -> (262144, 3)
    (32, 256, 32),  # three inputs -> (262144, 3)
    (16, 128, 64, 16),  # four inputs -> (2097152, 4)
    (4, 512, 8, 256),  # four inputs -> (4194304, 4)
)


class CartesianProdBenchmark(OperatorBenchmark):
    """GenericBenchmark whose shape source is the list of 1-D input sizes.

    The default shape file (dense tensor shapes) does not describe this op, so
    ``set_shapes`` substitutes the local case list and ``set_more_shapes``
    contributes nothing extra.
    """

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_CARTESIAN_PROD_BENCH_SIZES)

    def set_more_shapes(self):
        return []


def _case_fn(sizes, dtype):
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"inputs": list(sizes)},
        params={"n_inputs": len(sizes)},
        builder_args=(sizes,),
    )


def _build_inputs_fn(plan, dtype, device):
    sizes = plan.builder_args[0]
    tensors = [utils.generate_tensor_input(size, dtype, device) for size in sizes]
    return tensors, {}


@pytest.mark.cartesian_prod
def test_cartesian_prod():
    bench = CartesianProdBenchmark(
        op_name="cartesian_prod",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten.cartesian_prod,
        # KernelGen injects the candidate via --override; the
        # direct module callable may not exist until the op is merged, in which
        # case the benchmark falls back to the dispatcher reference.
        gems_op=getattr(flag_gems, "cartesian_prod", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
