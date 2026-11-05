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

from .accuracy_utils import gems_assert_close, init_seed

pytestmark = pytest.mark.native_layer_norm
DTYPES = [torch.float16, torch.bfloat16, torch.float32]
LIMITS = {
    torch.float16: (0.003, 0.003),
    torch.bfloat16: (0.03, 0.01),
    torch.float32: (2e-5, 3e-5),
}


def _oracle(x, normalized_shape, weight, bias, eps):
    dims = tuple(range(x.ndim - len(normalized_shape), x.ndim))
    z = x.cpu().double()
    mean = z.mean(dims, keepdim=True)
    centered = z - mean
    rstd = torch.rsqrt((centered * centered).mean(dims, keepdim=True) + eps)
    y = centered * rstd
    if weight is not None:
        y = y * weight.cpu().double()
    else:
        pass
    if bias is not None:
        y = y + bias.cpu().double()
    else:
        pass
    statistics_dtype = torch.float32 if flag_gems.vendor_name == "mthreads" else x.dtype
    return y.to(x.dtype), mean.to(statistics_dtype), rstd.to(statistics_dtype)


def _check(actual, expected, dtype):
    atol, _ = LIMITS[expected.dtype]
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    gems_assert_close(actual.cpu(), expected, expected.dtype, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mthreads_layer_norm_fp32_affine(dtype):
    init_seed(20261102)
    x = torch.randn((7, 128), device=flag_gems.device, dtype=dtype)
    w = torch.randn(128, device=x.device, dtype=torch.float32)
    b = torch.randn_like(w)
    result = flag_gems.native_layer_norm(x, (128,), w, b)
    for actual, expected in zip(result, _oracle(x, (128,), w, b, 1e-5)):
        _check(actual, expected, dtype)
