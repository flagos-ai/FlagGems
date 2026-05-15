"""
Accuracy tests for mHC (Manifold Constrained Hyper-Connection) operators.

Tests both mhc_post and mhc_pre against PyTorch reference implementations,
and optionally compares with TileLang implementations.
"""
from itertools import product

import pytest
import torch

import flag_gems
from flag_gems.fused.mhc.hc_head_fused_kernel import (
    hc_head_fused_kernel,
    hc_head_fused_kernel_ref,
)
from flag_gems.fused.mhc.hc_split_sinkhorn import (
    hc_split_sinkhorn,
    mhc_split_sinkhorn_torch_ref,
)
from flag_gems.fused.mhc.mhc_bwd import mhc_bwd, mhc_bwd_ref, sinkhorn_forward
from flag_gems.fused.mhc.mhc_post import mhc_post, mhc_post_ref
from flag_gems.fused.mhc.mhc_pre import mhc_pre, mhc_pre_ref


def generate_mhc_post_data(
    n: int, h: int, hc_mult: int = 4, device: str = flag_gems.device
):
    torch.manual_seed(42)
    x = torch.randn((n, h), dtype=torch.bfloat16, device=device)
    residual = torch.randn((n, hc_mult, h), dtype=torch.bfloat16, device=device)
    post_layer_mix = torch.randn((n, hc_mult, 1), dtype=torch.float32, device=device)
    comb_res_mix = torch.randn(
        (n, hc_mult, hc_mult), dtype=torch.float32, device=device
    )
    return dict(
        x=x, residual=residual, post_layer_mix=post_layer_mix, comb_res_mix=comb_res_mix
    )


MHC_POST_CONFIGS = list(
    product(
        [4096],  # n (num_tokens)
        [1280, 2560, 7168],  # h (hidden_size)
        [2, 4],  # hc_mult
    )
)


@pytest.mark.mhc_post
@pytest.mark.parametrize(
    "n, h, hc_mult",
    MHC_POST_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_POST_CONFIGS],
)
def test_mhc_post_vs_ref(n, h, hc_mult):
    """Test Triton mhc_post against PyTorch CPU reference."""
    data = generate_mhc_post_data(n, h, hc_mult=hc_mult)
    out_triton = mhc_post(**data)
    data_cpu = {k: v.cpu() for k, v in data.items()}
    out_ref = mhc_post_ref(**data_cpu)
    torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-2, atol=1e-2)


def generate_mhc_split_sinkhorn_data(
    batch: int,
    seqlen: int,
    hc_mult: int = 4,
    device: str = flag_gems.device,
):
    torch.manual_seed(42)
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn((batch, seqlen, mix_hc), dtype=torch.float32, device=device)
    hc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((mix_hc,), dtype=torch.float32, device=device) * 0.1
    return dict(mixes=mixes, hc_scale=hc_scale, hc_base=hc_base, hc_mult=hc_mult)


MHC_SPLIT_SINKHORN_CONFIGS = [
    (8, 16, 4),
    (32, 64, 4),
    (128, 128, 4),
    (256, 256, 4),
    (8, 16, 2),
    (32, 64, 2),
    (128, 128, 2),
    (256, 256, 2),
]


@pytest.mark.hc_split_sinkhorn_forward
@pytest.mark.parametrize(
    "batch, seqlen, hc_mult",
    MHC_SPLIT_SINKHORN_CONFIGS,
    ids=[f"b{b}_s{s}_hc{hc}" for b, s, hc in MHC_SPLIT_SINKHORN_CONFIGS],
)
def test_mhc_split_sinkhorn(batch, seqlen, hc_mult):
    """Test FlagGems split+sinkhorn implementation against DV."""
    data = generate_mhc_split_sinkhorn_data(batch, seqlen, hc_mult)
    pre_triton, post_triton, comb_triton = hc_split_sinkhorn(**data)
    pre_dv, post_dv, comb_dv = mhc_split_sinkhorn_torch_ref(**data)

    torch.testing.assert_close(pre_triton, pre_dv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(post_triton, post_dv, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(comb_triton, comb_dv, rtol=1e-4, atol=1e-4)


MHC_PRE_CONFIGS = list(
    product(
        [512, 1024, 2048, 8192],  # n
        [1280, 2560, 4096],  # hidden_size
        [2, 4],  # hc_mult
    )
)


def generate_mhc_pre_data(
    n: int,
    hc_mult: int,
    hidden_size: int,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 10,
    device: str = flag_gems.device,
):
    torch.manual_seed(42)
    hc_mult3 = hc_mult * 2 + hc_mult * hc_mult

    residual = (
        torch.randn((n, hc_mult, hidden_size), dtype=torch.float, device=device)
        .mul(1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
        .bfloat16()
    )
    fn = (
        torch.randn((hc_mult3, hc_mult, hidden_size), dtype=torch.float, device=device)
        * 1e-4
        * (1 + torch.arange(hc_mult, device=device).mul(0.01).view(1, -1, 1))
    ).flatten(1, 2)
    hc_scale = torch.randn((3,), dtype=torch.float, device=device) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=torch.float, device=device) * 0.1

    return dict(
        residual=residual,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        rms_eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_repeat=sinkhorn_repeat,
    )


@pytest.mark.mhc_pre
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_PRE_CONFIGS,
    ids=[f"n{n}_h{h}_hc_mult{hc_mult}" for n, h, hc_mult in MHC_PRE_CONFIGS],
)
def test_mhc_pre_vs_ref(n, hidden_size, hc_mult):
    """Test Triton mhc_pre against PyTorch CPU reference."""
    data = generate_mhc_pre_data(n, hc_mult, hidden_size)
    post_triton, comb_triton, li_triton = mhc_pre(**data)
    data_cpu = {
        k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data.items()
    }
    post_ref, comb_ref, li_ref = mhc_pre_ref(**data_cpu)

    torch.testing.assert_close(post_triton.cpu(), post_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(comb_triton.cpu(), comb_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(li_triton.cpu(), li_ref, rtol=1e-2, atol=1e-2)


MHC_BWD_CONFIGS = list(
    product(
        [256, 1024, 4096, 65536],  # seqlen
        [4],  # n_stream (optimized kernel only supports n_stream=4)
        [20],  # sinkhorn_iters
    )
)


def generate_mhc_bwd_data(
    seqlen: int,
    n_stream: int,
    sinkhorn_iters: int = 20,
    device: str = flag_gems.device,
):
    """Generate test data for mhc_bwd.

    Returns (R, dR) where R is Sinkhorn output and dR is upstream gradient.
    """
    torch.manual_seed(42)
    dist = torch.distributions.uniform.Uniform(0.0, 4.0)
    M = dist.sample((seqlen, n_stream, n_stream)).to(device)

    R, _P = sinkhorn_forward(M, iters=sinkhorn_iters)
    dR = torch.randn_like(R)

    return dict(R=R.detach(), dR=dR, n_stream=n_stream)


@pytest.mark.mhc_bwd
@pytest.mark.parametrize(
    "seqlen, n_stream, sinkhorn_iters",
    MHC_BWD_CONFIGS,
    ids=[f"seq{s}_ns{ns}_it{it}" for s, ns, it in MHC_BWD_CONFIGS],
)
def test_mhc_bwd_vs_ref(seqlen, n_stream, sinkhorn_iters):
    """Test Triton mhc_bwd against PyTorch CPU reference."""
    data = generate_mhc_bwd_data(seqlen, n_stream, sinkhorn_iters)
    R, dR = data["R"], data["dR"]

    out_triton = mhc_bwd(R, dR)
    out_ref = mhc_bwd_ref(R.cpu(), dR.cpu())

    torch.testing.assert_close(out_triton.cpu(), out_ref, rtol=1e-4, atol=1e-4)


MHC_HC_HEAD_FUSED_CONFIGS = [
    (256, 1280, 2),
    (256, 1280, 4),
    (512, 1280, 2),
    (512, 1280, 4),
    (512, 2560, 2),
    (512, 2560, 4),
    (1024, 2560, 2),
    (1024, 2560, 4),
    (2048, 4096, 2),
    (2048, 4096, 4),
    (4096, 1280, 2),
    (4096, 1280, 4),
]


def generate_hc_head_fused_data(
    n: int,
    hidden_size: int,
    hc_mult: int,
    dtype: torch.dtype,
    device: str = flag_gems.device,
):
    torch.manual_seed(42)
    hs_flat = torch.randn((n, hc_mult, hidden_size), dtype=dtype, device=device)
    fn = torch.randn(
        (hc_mult, hc_mult * hidden_size), dtype=torch.float32, device=device
    )
    hc_scale = torch.randn((1,), dtype=torch.float32, device=device) * 0.1
    hc_base = torch.randn((hc_mult,), dtype=torch.float32, device=device) * 0.1
    out = torch.empty((n, hidden_size), dtype=dtype, device=device)
    return dict(
        hs_flat=hs_flat,
        fn=fn,
        hc_scale=hc_scale,
        hc_base=hc_base,
        out=out,
        hidden_size=hidden_size,
        rms_eps=1e-6,
        hc_eps=1e-6,
        hc_mult=hc_mult,
    )


@pytest.mark.hc_head_fused_kernel
@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.float16, torch.bfloat16],
    ids=["fp32", "fp16", "bf16"],
)
@pytest.mark.parametrize(
    "n, hidden_size, hc_mult",
    MHC_HC_HEAD_FUSED_CONFIGS,
    ids=[f"n{n}_h{h}_hc{hc}" for n, h, hc in MHC_HC_HEAD_FUSED_CONFIGS],
)
def test_hc_head_fused_kernel_vs_ref(n, hidden_size, hc_mult, dtype):
    data = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)
    data_ref = generate_hc_head_fused_data(n, hidden_size, hc_mult, dtype=dtype)

    tol_map = {
        torch.float32: (1e-4, 1e-4),
        torch.float16: (2e-2, 2e-2),
        torch.bfloat16: (2e-2, 2e-2),
    }
    rtol, atol = tol_map[dtype]

    out_triton = hc_head_fused_kernel(**data)
    out_ref = hc_head_fused_kernel_ref(**data_ref)
    torch.testing.assert_close(out_triton, out_ref, rtol=rtol, atol=atol)
