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

IS_ENFLAME = flag_gems.vendor_name == "enflame"


def _batch_norm_stats_reference(inp, eps):
    """Independent per-channel reference built from raw tensor math.

    enflame GCU has no aten kernel for batch_norm_stats (not on CPU, and the
    on-device aten dispatch is the FlagGems implementation itself), so the
    reference is derived from the definition of the op instead: the per-channel
    mean and 1/sqrt(biased var + eps) accumulated in float32.
    """
    reduce_dims = (0,) + tuple(range(2, inp.dim()))
    inp_f32 = inp.to(torch.float32)
    ref_mean = inp_f32.mean(dim=reduce_dims)
    ref_var = inp_f32.var(dim=reduce_dims, unbiased=False)
    return ref_mean, 1.0 / torch.sqrt(ref_var + eps)


@pytest.mark.batch_norm_stats
@pytest.mark.parametrize(
    "shape",
    [
        (16, 3),
        (32, 32, 32),
        (8, 32, 224, 224),
        (2050, 16, 32, 32),
        (8, 16, 3, 224, 224),
    ],
)
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
def test_batch_norm_stats(shape, dtype):
    eps = 1e-5
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)

    if IS_ENFLAME:
        # Keep the enflame-specific fp32 reference: aten has no batch_norm_stats
        # kernel there and on-device aten dispatch would hit FlagGems itself.
        ref_mean, ref_invstd = _batch_norm_stats_reference(inp, eps)
    else:
        # torch.batch_norm_stats has no CPU kernel, so compute the reference
        # from the definition instead: batch norm normalises with the biased
        # (population) variance over every dim except the channel dim (dim 1).
        # Reduce in the to_reference dtype (float64 on CPU) and compare as
        # float32, the dtype the operator always returns.
        ref_inp = utils.to_reference(inp, True)
        reduce_dims = (0,) + tuple(range(2, ref_inp.dim()))
        ref_mean = ref_inp.mean(dim=reduce_dims)
        ref_var = ref_inp.var(dim=reduce_dims, correction=0)
        ref_invstd = 1.0 / torch.sqrt(ref_var + eps)

    res_mean, res_invstd = flag_gems.batch_norm_stats(inp, eps)

    # batch_norm_stats always returns float32 regardless of input dtype
    utils.gems_assert_close(res_mean, ref_mean, torch.float32)
    utils.gems_assert_close(res_invstd, ref_invstd, torch.float32)
