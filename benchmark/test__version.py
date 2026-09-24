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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_version`` starts with an underscore and ``pytest.mark`` refuses to
# generate a marker through attribute access for such names, so register it
# directly on the MarkGenerator to keep both ``@pytest.mark._version`` and
# ``-m _version`` working.
try:
    pytest.mark._version
except AttributeError:
    setattr(
        pytest.mark,
        "_version",
        MarkDecorator(Mark("_version", (), {}, _ispytest=True), _ispytest=True),
    )

# aten::_version(Tensor self) -> int reads the per-tensor version counter that
# every in-place mutation bumps. It is a pure O(1) metadata query whose measured
# latency is independent of the payload, so the benchmark shapes only control
# input allocation outside the timed region: they cover ranks 1-4 with
# representative element counts.
_VERSION_SHAPES = [
    (1,),
    (256,),
    (1024, 1024),
    (20, 320, 15),
    (4096, 4096),
    (64, 512, 512),
    (16, 256, 256, 16),
]


def _case_fn(shape, dtype):
    # One case per shape (the op takes a single tensor and no scalar params).
    del dtype
    yield base.BenchmarkCasePlan(
        shape={"input": tuple(shape)},
        params={},
        builder_args=(tuple(shape),),
    )


def _build_inputs_fn(plan, dtype, device):
    shape = plan.builder_args[0]
    inp = utils.generate_tensor_input(shape, dtype, device)
    return inp, {}


class VersionBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark with shapes tuned for the O(1) _version query."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path, default_shapes=_VERSION_SHAPES)


@pytest.mark._version
def test__version():
    bench = VersionBenchmark(
        op_name="_version",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._version,
        # ``flag_gems._version`` is the package version string (package
        # metadata), not a callable operator implementation, so the explicit
        # default is None: the candidate is resolved from the process-local
        # override injected by KernelGen, and without one the benchmark falls
        # back to the dispatcher route (which measures the native operator).
        gems_op=None,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
