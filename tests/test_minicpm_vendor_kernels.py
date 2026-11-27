# SPDX-License-Identifier: Apache-2.0
import math

import pytest
import torch

import flag_gems


@pytest.mark.flash_attn_varlen_func
@pytest.mark.skipif(flag_gems.vendor_name != "iluvatar", reason="BI-V150 decode kernel")
def test_iluvatar_paged_decode_partial_page():
    from flag_gems.ops.flash_decode_splitkv import flash_decode_splitkv

    device = flag_gems.device
    q = torch.randn(1, 16, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(3, 16, 2, 128, dtype=q.dtype, device=device)
    v = torch.randn_like(k)
    table = torch.tensor([[2, 0, 1]], dtype=torch.int32, device=device)
    lengths = torch.tensor([33], dtype=torch.int32, device=device)
    offsets = torch.tensor([0, 1], dtype=torch.int32, device=device)
    scale = 1.0 / math.sqrt(128)

    # Non-sequential pages and a one-token tail exercise both masking and reduction.
    keys = k.cpu()[[2, 0, 1]].reshape(48, 2, 128)[:33].float()
    values = v.cpu()[[2, 0, 1]].reshape(48, 2, 128)[:33].float()
    query = q.cpu().reshape(2, 8, 128).float()
    scores = torch.einsum("hgd,thd->hgt", query, keys) * scale
    expected = torch.einsum("hgt,thd->hgd", scores.softmax(-1), values)

    out = torch.empty_like(q)
    flash_decode_splitkv(q, k, v, out, offsets, lengths, table, scale, block_n=32)
    torch.testing.assert_close(
        out.cpu().float(), expected.reshape_as(out), atol=0.02, rtol=0.02
    )


@pytest.mark.mm
@pytest.mark.skipif(flag_gems.vendor_name != "metax", reason="C500 projection kernel")
def test_metax_projection_preserves_strided_output():
    from flag_gems.runtime.backend._metax.ops.mm import mm_out

    device = flag_gems.device
    a = torch.randn(3, 2048, dtype=torch.bfloat16, device=device) / math.sqrt(2048)
    weight = torch.randn(2560, 2048, dtype=a.dtype, device=device)
    storage = torch.full((3, 2561), float("nan"), device=device, dtype=a.dtype)
    out = storage[:, :2560]
    expected = a.cpu().float() @ weight.cpu().float().t()

    result = mm_out(a, weight.t(), out=out)
    assert result is out
    assert storage[:, -1].isnan().all().item()
    torch.testing.assert_close(out.cpu().float(), expected, atol=0.03, rtol=0.03)
