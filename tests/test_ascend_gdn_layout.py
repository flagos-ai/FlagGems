import pytest
import torch

import flag_gems

pytestmark = pytest.mark.skipif(
    flag_gems.device != "npu", reason="Ascend GDN implementation"
)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gdn_strided_qkv_and_indexed_state(packed, dtype):
    from flag_gems.runtime.backend._ascend.fla import (
        fused_sigmoid_gating_delta_rule_update,
    )

    device = flag_gems.device
    n, h, hv, kdim, vdim = 4, 2, 6, 16, 32
    data = torch.randn(n, 2 * h * kdim + hv * vdim, device=device, dtype=dtype) * 0.1
    shape = (1, n) if packed else (n, 1)
    q = data[:, : h * kdim].view(*shape, h, kdim)
    k = data[:, h * kdim : 2 * h * kdim].view(*shape, h, kdim)
    v = data[:, 2 * h * kdim :].view(*shape, hv, vdim)
    state = torch.randn(n + 1, hv, kdim, vdim, device=device, dtype=dtype) * 0.1
    before = state.clone()
    indices = torch.tensor([-1, 3, 0, 2], device=device, dtype=torch.int32)
    a = torch.randn(n, hv, device=device, dtype=dtype) * 0.1
    b = torch.randn_like(a)
    A = torch.randn(hv, device=device) * 0.1
    bias = torch.randn(hv, device=device, dtype=dtype) * 0.1
    expected = []
    expected_state = before.clone()
    for row, index in enumerate([-1, 3, 0, 2]):
        qq = q.reshape(n, h, kdim)[row].float().repeat_interleave(hv // h, dim=0)
        kk = k.reshape(n, h, kdim)[row].float().repeat_interleave(hv // h, dim=0)
        qq = qq / (qq.square().sum(-1).sqrt()[:, None] + 1e-6) * kdim**-0.5
        kk = kk / (kk.square().sum(-1).sqrt()[:, None] + 1e-6)
        s = torch.zeros_like(before[0]).float() if index < 0 else before[index].float()
        g = torch.exp(
            -torch.exp(A) * torch.nn.functional.softplus(a[row].float() + bias.float())
        )
        s = s * g[:, None, None]
        delta = (v.reshape(n, hv, vdim)[row].float() - (s * kk[:, :, None]).sum(1)) * b[
            row
        ].float().sigmoid()[:, None]
        s = s + kk[:, :, None] * delta[:, None, :]
        expected.append((s * qq[:, :, None]).sum(1).to(dtype))
        if index >= 0:
            expected_state[index] = s.to(dtype)
    result = fused_sigmoid_gating_delta_rule_update(
        A,
        a,
        bias,
        1.0,
        20.0,
        q,
        k,
        v,
        b,
        state,
        indices,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=torch.arange(n + 1, device=device, dtype=torch.int32)
        if packed
        else None,
    )
    torch.testing.assert_close(
        result, torch.stack(expected).view_as(v), rtol=2e-2, atol=2e-3
    )
    torch.testing.assert_close(state, expected_state, rtol=2e-2, atol=2e-3)
    torch.testing.assert_close(state[1], before[1], rtol=0, atol=0)
