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


@pytest.mark.huber_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss(shape, dtype, reduction, delta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    ref_out = torch.ops.aten.huber_loss(ref_inp, ref_target, reduction, float(delta))
    with flag_gems.use_gems():
        res_out = torch.ops.aten.huber_loss(inp, target, reduction, float(delta))

    reduce_dim = target.numel() if reduction != 0 else shape[-1]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)


@pytest.mark.huber_loss
@pytest.mark.parametrize("shape", [(2, 3), (128, 256), (512, 512)])
@pytest.mark.parametrize("dtype", utils.FLOAT_DTYPES)
@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("delta", [0.5, 1.0, 2.0])
def test_huber_loss_out(shape, dtype, reduction, delta):
    inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    target = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    ref_inp = utils.to_reference(inp, True)
    ref_target = utils.to_reference(target, True)

    out_shape = shape if reduction == 0 else ()
    ref_out = torch.empty(out_shape, dtype=ref_inp.dtype, device=ref_inp.device)
    torch.ops.aten.huber_loss.out(
        ref_inp, ref_target, reduction, float(delta), out=ref_out
    )
    with flag_gems.use_gems():
        res_out = torch.empty(out_shape, dtype=dtype, device=flag_gems.device)
        torch.ops.aten.huber_loss.out(inp, target, reduction, float(delta), out=res_out)

    reduce_dim = target.numel() if reduction != 0 else shape[-1]
    utils.gems_assert_close(res_out, ref_out, dtype, reduce_dim=reduce_dim)
