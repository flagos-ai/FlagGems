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

from . import base, consts


# TODO(Qiming): Extract this to a base class
class NormBenchmark(base.GenericBenchmark):
    def set_more_shapes(self):
        return [
            (16, 16, 64),
            (16, 16, 1024),
            (16, 16, 4098),
            (1, 8, 4, 4),
            (16, 8, 128, 128),
            (256, 512),
            (4096, 256),
            (4096, 512),
            (2048, 1024),
            (1024, 2048),
        ]


# The production call feeds a [65536, 1, 1152] activation with a (1152,)
# normalized shape at eps 1e-6; every other core shape drops the trailing
# dimension and keeps the default eps.
REAL_CALL_OVERRIDES = {
    (65536, 1, 1152): ((1152,), 1e-6),
}


def input_fn(shape, dtype, device):
    normalized_shape, eps = REAL_CALL_OVERRIDES.get(shape, (shape[1:], 1e-5))
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    yield inp, normalized_shape, weight, bias, eps


@pytest.mark.native_layer_norm
def test_native_layer_norm():
    bench = NormBenchmark(
        op_name="native_layer_norm",
        input_fn=input_fn,
        torch_op=torch.ops.aten.native_layer_norm.default,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


@pytest.mark.layer_norm
def test_layer_norm():
    bench = NormBenchmark(
        op_name="layer_norm",
        input_fn=input_fn,
        torch_op=torch.layer_norm,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()


def input_fn_backward(shape, dtype, device):
    grad_out = torch.randn(shape, dtype=dtype, device=device)
    inp = torch.randn(shape, dtype=dtype, device=device)
    normalized_shape = shape[1:]
    # mean and rstd are always float32 for native_layer_norm_backward
    mean = torch.randn(shape[0], dtype=torch.float32, device=device)
    rstd = torch.randn(shape[0], dtype=torch.float32, device=device)
    weight = torch.randn(normalized_shape, dtype=dtype, device=device)
    bias = torch.randn(normalized_shape, dtype=dtype, device=device)
    output_mask = [True, True, True]
    yield grad_out, inp, normalized_shape, mean, rstd, weight, bias, output_mask


@pytest.mark.layer_norm_backward
def test_layer_norm_backward():
    bench = NormBenchmark(
        op_name="layer_norm_backward",
        input_fn=input_fn_backward,
        torch_op=torch.ops.aten.native_layer_norm_backward,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
