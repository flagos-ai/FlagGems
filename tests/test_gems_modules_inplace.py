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


@pytest.mark.rms_norm
def test_gems_rms_forward_inplace_many_rows():
    """Fused add + RMS norm is in-place; more rows than AI cores must be correct.

    An Ascend launch with more programs than AI cores runs in several program
    waves, and the later waves used to overwrite the row the first wave was
    still writing (issue #5624).
    """
    from flag_gems.modules.normalization import gems_rms_forward

    torch.manual_seed(42)
    rows, cols = 1024, 512
    x = torch.randn((rows, cols), dtype=torch.float32, device=flag_gems.device)
    residual = torch.randn((rows, cols), dtype=torch.float32, device=flag_gems.device)
    weight = torch.randn((cols,), dtype=torch.float32, device=flag_gems.device)

    out, new_residual = gems_rms_forward(x.clone(), residual.clone(), weight, 1e-5)

    # Independent CPU reference (no flag_gems kernels), moved back for compare.
    ref_x, ref_residual_in, ref_weight = x.cpu(), residual.cpu(), weight.cpu()
    ref_residual = ref_x + ref_residual_in
    ref_out = (
        ref_residual
        * torch.rsqrt(ref_residual.pow(2).mean(-1, keepdim=True) + 1e-5)
        * ref_weight
    )
    utils.gems_assert_close(new_residual, ref_residual.to(new_residual.device), torch.float32)
    utils.gems_assert_close(out, ref_out.to(out.device), torch.float32)


@pytest.mark.apply_rotary_pos_emb
@pytest.mark.skipif(
    flag_gems.vendor_name != "ascend",
    reason="pins the Ascend single-wave in-place launch path (#5624)",
)
def test_gems_rope_inplace_matches_out_of_place():
    """In-place must rotate exactly like out-of-place.

    On Ascend an in-place kernel launched with more programs than the number of
    AI cores runs in several program waves, and the later waves used to
    overwrite the rows the earlier waves were still writing, corrupting the
    first tile (issue #5624).
    """
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    torch.manual_seed(42)
    batch, seq_len, heads, head_dim = 1, 129, 32, 128
    query = torch.randn(
        (batch, seq_len, heads, head_dim), dtype=torch.float32, device=flag_gems.device
    )
    key = torch.randn(
        (batch, seq_len, heads, head_dim), dtype=torch.float32, device=flag_gems.device
    )
    cos = torch.rand(seq_len, head_dim // 2, dtype=torch.float32, device=flag_gems.device)
    sin = torch.rand(seq_len, head_dim // 2, dtype=torch.float32, device=flag_gems.device)

    ref_q, ref_k = gems_rope_forward(query.clone(), key.clone(), cos, sin, inplace=False)
    q_in, k_in = query.clone(), key.clone()
    out_q, out_k = gems_rope_forward(q_in, k_in, cos, sin, inplace=True)

    # In-place must write through the caller's buffers ...
    assert out_q.data_ptr() == q_in.data_ptr()
    assert out_k.data_ptr() == k_in.data_ptr()
    # ... and produce the same numbers as the out-of-place path.
    utils.gems_assert_close(q_in, ref_q, torch.float32)
    utils.gems_assert_close(k_in, ref_k, torch.float32)
