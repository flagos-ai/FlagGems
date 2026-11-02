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

from . import base, consts, utils


def feature_alpha_dropout_input_fn(shape, dtype, device):
    input = utils.generate_tensor_input(shape, dtype, device)
    yield input, 0.5, True


class FeatureAlphaDropoutBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = [
            (2, 3),
            (4, 8, 16),
            (8, 16, 32, 32),
            (16, 32, 64, 64),
            (32, 64, 64, 64),
            (32, 64, 128, 128),
            (4, 8, 16, 16, 16),
        ]


@pytest.mark.feature_alpha_dropout
def test_feature_alpha_dropout():
    benchmark = FeatureAlphaDropoutBenchmark(
        input_fn=feature_alpha_dropout_input_fn,
        op_name="feature_alpha_dropout",
        torch_op=torch.ops.aten.feature_alpha_dropout,
        dtypes=consts.FLOAT_DTYPES,
    )
    benchmark.run()
