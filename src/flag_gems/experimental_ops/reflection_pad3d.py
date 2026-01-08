import torch
import triton
import triton.language as tl


@triton.jit
def reflection_pad3d_kernel(
    in_ptr,
    out_ptr,
    N,
    C,
    Di,
    Hi,
    Wi,
    Do,
    Ho,
    Wo,
    pad_w_left,
    pad_w_right,
    pad_h_top,
    pad_h_bottom,
    pad_d_front,
    pad_d_back,
    in_sN,
    in_sC,
    in_sD,
    in_sH,
    in_sW,
    out_sN,
    out_sC,
    out_sD,
    out_sH,
    out_sW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    tmp = offs.to(tl.int64)

    w = tmp % Wo
    tmp = tmp // Wo
    h = tmp % Ho
    tmp = tmp // Ho
    d = tmp % Do
    tmp = tmp // Do
    c = tmp % C
    n = tmp // C

    # Reflection mapping for W
    w_rel = w - pad_w_left
    m_w = Wi - 1
    two_m_w = 2 * m_w
    denom_w = tl.where(m_w > 0, two_m_w, 1)
    r_w = w_rel % denom_w
    r_w = tl.where(r_w < 0, r_w + denom_w, r_w)
    w_in = tl.where(m_w > 0, m_w - tl.abs(m_w - r_w), 0)

    # Reflection mapping for H
    h_rel = h - pad_h_top
    m_h = Hi - 1
    two_m_h = 2 * m_h
    denom_h = tl.where(m_h > 0, two_m_h, 1)
    r_h = h_rel % denom_h
    r_h = tl.where(r_h < 0, r_h + denom_h, r_h)
    h_in = tl.where(m_h > 0, m_h - tl.abs(m_h - r_h), 0)

    # Reflection mapping for D
    d_rel = d - pad_d_front
    m_d = Di - 1
    two_m_d = 2 * m_d
    denom_d = tl.where(m_d > 0, two_m_d, 1)
    r_d = d_rel % denom_d
    r_d = tl.where(r_d < 0, r_d + denom_d, r_d)
    d_in = tl.where(m_d > 0, m_d - tl.abs(m_d - r_d), 0)

    in_offset = (
        (n * in_sN) + (c * in_sC) + (d_in * in_sD) + (h_in * in_sH) + (w_in * in_sW)
    )
    out_offset = (
        (n * out_sN) + (c * out_sC) + (d * out_sD) + (h * out_sH) + (w * out_sW)
    )

    val = tl.load(in_ptr + in_offset, mask=mask, other=0)
    tl.store(out_ptr + out_offset, val, mask=mask)


def _reflection_pad3d_launch(input: torch.Tensor, padding, out: torch.Tensor = None):
    assert input.dim() == 5, "reflection_pad3d expects a 5D tensor (N, C, D, H, W)"
    assert (
        len(padding) == 6
    ), "padding must be a 6-element tuple: (pl, pr, pt, pb, pf, pbk)"
    pl, pr, pt, pb, pf, pbk = map(int, padding)

    N, C, Di, Hi, Wi = input.shape
    assert all(
        p >= 0 for p in (pl, pr, pt, pb, pf, pbk)
    ), "All paddings must be non-negative"
    assert Wi > 0 and Hi > 0 and Di > 0, "Input spatial dimensions must be > 0"
    assert pl < Wi and pr < Wi, "W paddings must be less than input W"
    assert pt < Hi and pb < Hi, "H paddings must be less than input H"
    assert pf < Di and pbk < Di, "D paddings must be less than input D"

    Do = Di + pf + pbk
    Ho = Hi + pt + pb
    Wo = Wi + pl + pr

    if out is None:
        out = torch.empty((N, C, Do, Ho, Wo), device=input.device, dtype=input.dtype)
    else:
        assert out.shape == (N, C, Do, Ho, Wo), "out tensor has incorrect shape"
        assert (
            out.device == input.device
        ), "out tensor must be on the same device as input"
        assert out.dtype == input.dtype, "out tensor must have the same dtype as input"

    in_sN, in_sC, in_sD, in_sH, in_sW = input.stride()
    out_sN, out_sC, out_sD, out_sH, out_sW = out.stride()

    n_elements = out.numel()
    if n_elements == 0:
        return out

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    reflection_pad3d_kernel[grid](
        input,
        out,
        N,
        C,
        Di,
        Hi,
        Wi,
        Do,
        Ho,
        Wo,
        pl,
        pr,
        pt,
        pb,
        pf,
        pbk,
        in_sN,
        in_sC,
        in_sD,
        in_sH,
        in_sW,
        out_sN,
        out_sC,
        out_sD,
        out_sH,
        out_sW,
        n_elements,
        BLOCK_SIZE=1024,
    )
    return out


def reflection_pad3d(input: torch.Tensor, padding):
    return _reflection_pad3d_launch(input, padding, out=None)


def reflection_pad3d_out(input: torch.Tensor, padding, out: torch.Tensor):
    return _reflection_pad3d_launch(input, padding, out=out)
