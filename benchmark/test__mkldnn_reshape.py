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

# aten::_mkldnn_reshape consumes and returns oneDNN ("mkldnn") opaque tensors,
# which exist on the CPU backend only (aten::to_mkldnn is registered for CPU and
# fails for CUDA). Every case therefore builds its operand on CPU and hands the
# same opaque tensor type to the native reference and to the injected candidate.
MKLDNN_RESHAPE_EXTRA_SHAPES = [
    (64, 64),
    (1024, 1024),
    (4096, 4096),
    (64, 512, 512),
    (128, 256, 256),
    (20, 320, 15),
]

# The five mkldnn-representable families. These are CPU fixtures, so GPU
# capability flags do not apply; int8/uint8 have real mkldnn reorders.
MKLDNN_RESHAPE_DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.int8,
    torch.uint8,
]


def _target_shape(shape):
    """Axis-reversed reshape: numel-preserving for every rank used here."""
    return tuple(reversed(shape))


def _case_fn(shape, dtype):
    del dtype
    # Tensor-free: only JSON-safe metadata plus private builder_args.
    yield base.BenchmarkCasePlan(
        shape={"input": list(shape)},
        params={"shape": list(_target_shape(shape))},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    del device
    (shape,) = plan.builder_args
    # This payload is allocated outside timing and never read, so torch.empty on
    # CPU is enough and works for every mkldnn-representable dtype. Flat
    # positional arguments with a trailing kwargs dict, as unpack_to_args_kwargs
    # expects.
    inp = torch.empty(shape, dtype=dtype).to_mkldnn()
    return inp, {"shape": list(_target_shape(shape))}


class MkldnnReshapeBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark over the shared shapes plus mkldnn extras."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(shape_file_path)
        # Only rank 0 is illegal for this operator (a rank-0 operand has no
        # mkldnn reorder primitive), so that single rank is filtered and the
        # shared core/comprehensive shapes are deduplicated with the extras.
        shapes = [
            tuple(shape) for shape in list(self.shapes) + MKLDNN_RESHAPE_EXTRA_SHAPES
        ]
        self.shapes = list(dict.fromkeys(s for s in shapes if len(s) > 0))


@pytest.mark.mkldnn_reshape
def test__mkldnn_reshape():
    bench = MkldnnReshapeBenchmark(
        op_name="_mkldnn_reshape",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._mkldnn_reshape,
        gems_op=getattr(flag_gems, "_mkldnn_reshape", None),
        dtypes=MKLDNN_RESHAPE_DTYPES,
    )
    bench.run()
