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

from . import accuracy_utils as utils
from .conftest import QUICK_MODE

if QUICK_MODE:
    # QUICK_MODE local-dev fast path: single dtype to keep the smoke run short.
    FLOAT_DTYPES = [torch.float16]
    # (N, C, H_in, W_in, H_out, W_out, align_corners)
    PARAMS_BWD = [(1, 3, 16, 16, 8, 8, False)]
else:
    # original: utils.FLOAT_DTYPES (3 dtypes), reduced to 2 to avoid CI timeout
    FLOAT_DTYPES = utils.PRIMARY_FLOAT_DTYPES
    # (N, C, H_in, W_in, H_out, W_out, align_corners): mix of upsample and
    # downsample, C=1 GEMM edge cases, and a large NC batch to hit both the
    # fused and 2-pass Triton paths.
    PARAMS_BWD = [
        (1, 3, 16, 16, 8, 8, False),
        (2, 4, 8, 8, 16, 16, False),
        (1, 3, 32, 32, 10, 10, False),
        (1, 3, 16, 16, 8, 8, True),
        (1, 3, 8, 8, 16, 16, True),
        (2, 64, 32, 32, 16, 16, False),
        (1, 1, 64, 64, 16, 16, False),
        (4, 16, 64, 128, 32, 64, False),
    ]


def upsample_bilinear2d_aa_backward_call(grad, input_size, align_corners):
    orig_shape = tuple(input_size)
    n, c, in_h, in_w = orig_shape

    shape_4d = (n, c, in_h, in_w)
    out_h = grad.shape[-2]
    out_w = grad.shape[-1]

    grad_4d = grad.reshape(n, c, out_h, out_w)

    out = torch.ops.aten._upsample_bilinear2d_aa_backward(
        grad_4d,
        [out_h, out_w],
        list(shape_4d),
        align_corners,
        None,
        None,
    )

    return out.reshape(orig_shape)


@pytest.mark.upsample_bilinear2d_aa_backward
@pytest.mark.parametrize("N,C,H_in,W_in,H_out,W_out,align_corners", PARAMS_BWD)
@pytest.mark.parametrize("dtype", FLOAT_DTYPES)
@pytest.mark.skipif(
    flag_gems.vendor_name == "tsingmicro",
    reason="Issues #3861: some ops hang in op tests",
)
def test_upsample_bilinear2d_aa_backward(
    N, C, H_in, W_in, H_out, W_out, align_corners, dtype
):
    shape = (N, C, H_in, W_in)

    grad_shape = (N, C, H_out, W_out)

    res_grad = torch.randn(
        grad_shape,
        dtype=torch.float32,
        device=flag_gems.device,
    )
    ref_grad = utils.to_reference(res_grad)

    ref_out = upsample_bilinear2d_aa_backward_call(
        ref_grad,
        shape,
        align_corners,
    ).to(dtype)

    with flag_gems.use_gems():
        res_out = upsample_bilinear2d_aa_backward_call(
            res_grad.to(dtype),
            shape,
            align_corners,
        )

    assert res_out.shape == shape

    # dtype-specific tolerance: bilinear AA backward accumulates a small filter
    # window, so half precision needs a looser absolute tolerance.
    if dtype == torch.float32:
        atol = 1e-4
    elif dtype == torch.float16:
        atol = 3e-3
    else:  # bfloat16
        atol = 2e-2

    utils.gems_assert_close(res_out, ref_out, dtype, atol=atol)
