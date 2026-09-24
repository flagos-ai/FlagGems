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

from .generated_operator_utils import OperatorBenchmark
from . import base, consts, utils

# ``_new_zeros_with_same_feature_meta`` starts with an underscore, and
# ``pytest.mark`` refuses to generate a marker via attribute access for such
# names. Register it directly on the MarkGenerator so ``-m
# _new_zeros_with_same_feature_meta`` works.
setattr(
    pytest.mark,
    "_new_zeros_with_same_feature_meta",
    MarkDecorator(
        Mark("_new_zeros_with_same_feature_meta", (), {}, _ispytest=True),
        _ispytest=True,
    ),
)

# aten::_new_zeros_with_same_feature_meta allocates a zero tensor whose shape
# is ``self.shape[:self_num_batch_dims] + other.shape`` and whose dtype follows
# ``other``, so the benchmark measures the cost of that zero allocation plus the
# feature-meta bookkeeping. Every case allocates a sizable output so the timing
# reflects actual device allocation rather than pure dispatch overhead; the
# inputs themselves are kept comparatively small.
_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES = [
    ((4,), (1024, 1024), 1),
    ((16,), (256, 256), 1),
    ((64,), (512, 512), 1),
    ((32,), (1024, 64), 1),
    ((4, 16), (1024, 64), 2),
    ((8, 32), (512, 128), 2),
    ((2,), (20, 320, 15), 1),
    ((8, 16), (256, 256), 2),
    ((16, 8, 4), (128, 128), 3),
    ((2, 2, 2, 2), (64, 64), 4),
]


def _case_fn(shape, dtype):
    del dtype
    self_shape, other_shape, self_num_batch_dims = shape
    yield base.BenchmarkCasePlan(
        shape={"self": self_shape, "other": other_shape},
        params={"self_num_batch_dims": self_num_batch_dims},
        builder_args=(shape,),
    )


def _build_inputs_fn(plan, dtype, device):
    self_shape, other_shape, self_num_batch_dims = plan.builder_args[0]
    self_inp = utils.generate_tensor_input(self_shape, dtype, device)
    other_inp = utils.generate_tensor_input(other_shape, dtype, device)
    return self_inp, other_inp, {"self_num_batch_dims": self_num_batch_dims}


class NewZerosWithSameFeatureMetaBenchmark(OperatorBenchmark):
    """Two-phase GenericBenchmark restricted to the op's allocation shapes."""

    def set_shapes(self, shape_file_path=None):
        super().set_shapes(
            shape_file_path, default_shapes=_NEW_ZEROS_WITH_SAME_FEATURE_META_CASES
        )


@pytest.mark._new_zeros_with_same_feature_meta
def test__new_zeros_with_same_feature_meta():
    bench = NewZerosWithSameFeatureMetaBenchmark(
        op_name="_new_zeros_with_same_feature_meta",
        case_fn=_case_fn,
        build_inputs_fn=_build_inputs_fn,
        torch_op=torch.ops.aten._new_zeros_with_same_feature_meta,
        gems_op=getattr(flag_gems, "_new_zeros_with_same_feature_meta", None),
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
