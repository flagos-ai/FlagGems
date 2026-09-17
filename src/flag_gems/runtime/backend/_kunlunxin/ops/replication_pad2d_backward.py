import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)


@triton.jit
def _replication_pad2d_backward_bulk_kernel(
    go_ptr,
    gi_ptr,
    OW,
    W,
    H_2,
    pt,
    pl,
    OHW,
    HW,
    CPW: tl.constexpr,
    NEED_MASK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_id = pid // CPW
    chunk = pid % CPW
    nc = row_id // H_2
    ih = row_id % H_2
    iw = chunk * BLOCK + tl.arange(0, BLOCK)
    oh = pt + 1 + ih
    ow = pl + iw
    out_base = nc * OHW + oh * OW
    in_base = nc * HW + (1 + ih) * W
    if NEED_MASK:
        mask = iw < W
        v = tl.load(go_ptr + out_base + ow, mask=mask, other=0.0)
        tl.store(gi_ptr + in_base + iw, v, mask=mask)
    else:
        v = tl.load(go_ptr + out_base + ow)
        tl.store(gi_ptr + in_base + iw, v)


@triton.jit
def _replication_pad2d_backward_bulk_contig_kernel(
    go_ptr,
    gi_ptr,
    OW,
    W,
    H_2,
    pt,
    pl,
    OHW,
    HW,
    total,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    c = tl.program_id(1)
    off = c * BLOCK + tl.arange(0, BLOCK)
    mask = off < total
    r = off // W
    iw = off - r * W
    v = tl.load(go_ptr + nc * OHW + (pt + 1 + r) * OW + pl + iw, mask=mask, other=0.0)
    tl.store(gi_ptr + nc * HW + W + off, v, mask=mask)


@triton.jit
def _replication_pad2d_backward_row_edge_kernel(
    go_ptr,
    gi_ptr,
    OW,
    W,
    H,
    pt,
    pl,
    pr,
    pb,
    OHW,
    HW,
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    nc = tl.program_id(0)
    rid = tl.program_id(1)
    cg = tl.program_id(2)
    iw = cg * BLOCK + tl.arange(0, BLOCK)
    mask = iw < W

    ih = tl.where(rid == 0, 0, H - 1)
    lo_r = tl.where(rid == 0, 0, pt + H - 1)
    cnt_r = tl.where(rid == 0, pt + 1, pb + 1)

    lo_c_raw = tl.where(iw == 0, 0, tl.where(iw == W - 1, pl + W - 1, pl + iw))
    lo_c = tl.minimum(tl.maximum(lo_c_raw, 0), OW - 1)
    cnt_c = tl.where(
        iw == 0,
        tl.maximum(pl + 1, 0),
        tl.where(
            iw == W - 1,
            tl.where(lo_c_raw >= OW, 0, OW - tl.maximum(lo_c_raw, 0)),
            tl.where((lo_c_raw >= 0) & (lo_c_raw < OW), 1, 0),
        ),
    )

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(MAXG):
        rr = tl.minimum(r, tl.maximum(cnt_r - 1, 0))
        out_base = nc * OHW + (lo_r + rr) * OW
        for c in tl.static_range(MAXG):
            cc = tl.minimum(c, tl.maximum(cnt_c - 1, 0))
            v = tl.load(go_ptr + out_base + lo_c + cc, mask=mask, other=0.0).to(
                tl.float32
            )
            acc += tl.where((r < cnt_r) & (c < cnt_c), v, 0.0)
    in_base = nc * HW + ih * W
    tl.store(gi_ptr + in_base + iw, acc, mask=mask)


@triton.jit
def _replication_pad2d_backward_col_edge_kernel(
    go_ptr,
    gi_ptr,
    OW,
    W,
    H_2,
    pt,
    pl,
    pr,
    OHW,
    HW,
    MAXG: tl.constexpr,
    P: tl.constexpr,
):
    nc = tl.program_id(0)
    cid = tl.program_id(1)
    rg = tl.program_id(2)
    rr = rg * P + tl.arange(0, P)
    mask = rr < H_2
    ih = 1 + rr

    lo_c_raw = tl.where(cid == 0, 0, pl + W - 1)
    lo_c = tl.minimum(tl.maximum(lo_c_raw, 0), OW - 1)
    cnt_c = tl.where(
        cid == 0,
        tl.maximum(pl + 1, 0),
        tl.where(lo_c_raw >= OW, 0, OW - tl.maximum(lo_c_raw, 0)),
    )

    out_base = nc * OHW + (pt + ih) * OW
    acc = tl.zeros((P,), dtype=tl.float32)
    for c in tl.static_range(MAXG):
        cc = tl.minimum(c, tl.maximum(cnt_c - 1, 0))
        v = tl.load(go_ptr + out_base + lo_c + cc, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(c < cnt_c, v, 0.0)
    in_base = nc * HW + ih * W
    tl.store(gi_ptr + in_base + tl.where(cid == 0, 0, W - 1), acc, mask=mask)


@triton.jit
def _replication_pad2d_backward_colfold_kernel(
    go_ptr,
    cf_ptr,
    OW,
    W,
    pl,
    pr,
    OH,
    total,
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    iw = o % W
    rest = o // W
    oh = rest % OH
    nc = rest // OH

    lo_raw = tl.where(iw == 0, 0, tl.where(iw == W - 1, pl + W - 1, pl + iw))
    lo = tl.minimum(tl.maximum(lo_raw, 0), OW - 1)
    cnt = tl.where(
        iw == 0,
        tl.where(W == 1, OW, tl.maximum(pl + 1, 0)),
        tl.where(
            iw == W - 1,
            tl.where(lo_raw >= OW, 0, OW - tl.maximum(lo_raw, 0)),
            tl.where((lo_raw >= 0) & (lo_raw < OW), 1, 0),
        ),
    )

    out_base = nc * OH * OW + oh * OW
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for c in tl.static_range(MAXG):
        cc = tl.minimum(c, tl.maximum(cnt - 1, 0))
        v = tl.load(go_ptr + out_base + lo + cc, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(c < cnt, v, 0.0)
    tl.store(cf_ptr + o, acc, mask=mask)


@triton.jit
def _replication_pad2d_backward_rowfold_kernel(
    cf_ptr,
    gi_ptr,
    W,
    H,
    pt,
    pb,
    OH,
    total,
    MAXG: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    o = pid * BLOCK + tl.arange(0, BLOCK)
    mask = o < total

    iw = o % W
    rest = o // W
    ih = rest % H
    nc = rest // H

    lo_raw = tl.where(ih == 0, 0, tl.where(ih == H - 1, pt + H - 1, pt + ih))
    lo = tl.minimum(tl.maximum(lo_raw, 0), OH - 1)
    cnt = tl.where(
        ih == 0,
        tl.where(H == 1, OH, tl.maximum(pt + 1, 0)),
        tl.where(
            ih == H - 1,
            tl.where(lo_raw >= OH, 0, OH - tl.maximum(lo_raw, 0)),
            tl.where((lo_raw >= 0) & (lo_raw < OH), 1, 0),
        ),
    )

    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for r in tl.static_range(MAXG):
        rr = tl.minimum(r, tl.maximum(cnt - 1, 0))
        in_base = nc * OH * W + (lo + rr) * W
        v = tl.load(cf_ptr + in_base + iw, mask=mask, other=0.0).to(tl.float32)
        acc += tl.where(r < cnt, v, 0.0)
    tl.store(gi_ptr + o, acc, mask=mask)


def _normalize_padding(padding):
    if isinstance(padding, torch.Tensor):
        return tuple(int(p) for p in padding.tolist())
    if isinstance(padding, int):
        return (padding, padding, padding, padding)
    if not isinstance(padding, (tuple, list)) or len(padding) != 4:
        raise ValueError(
            "padding must be a sequence of 4 integers: "
            "(pad_left, pad_right, pad_top, pad_bottom)"
        )
    return tuple(int(p) for p in padding)


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    pl, pr, pt, pb = _normalize_padding(padding)

    is_3d = self.ndim == 3
    x = self.contiguous()
    if is_3d:
        x = x.unsqueeze(0)
    go = grad_output.contiguous()
    if is_3d:
        go = go.unsqueeze(0)

    N, C, H, W = x.shape
    OH = H + pt + pb
    OW = W + pl + pr
    if tuple(go.shape) != (N, C, OH, OW):
        raise ValueError(
            f"grad_output shape {tuple(go.shape)} does not match "
            f"expected {(N, C, OH, OW)}"
        )

    NC = N * C

    if NC * H * W == 0:
        res = torch.zeros(N, C, H, W, device=x.device, dtype=x.dtype)
        if out is not None:
            torch.ops.aten._copy_from(res.view(self.shape), out)
            return out
        return res.view(self.shape)

    if pl == 0 and pr == 0 and pt == 0 and pb == 0:
        res = go.view(N, C, H, W)
        if out is not None:
            torch.ops.aten._copy_from(res, out)
            return out
        return res.view(self.shape)

    gi = torch.empty(N, C, H, W, device=x.device, dtype=x.dtype)
    OHW = OH * OW
    HW = H * W
    with torch_device_fn.device(x.device):
        if H == 1 or W == 1:
            if W == 1 and H > 1:
                t = go.sum(dim=-1)
                _replication_pad2d_backward_rowfold_kernel[
                    (triton.cdiv(NC * H * 1, 256),)
                ](
                    t.reshape(NC, OH, 1),
                    gi,
                    1,
                    H,
                    pt,
                    pb,
                    OH,
                    NC * H * 1,
                    MAXG=max(pt + 1, pb + 1, 1),
                    BLOCK=256,
                )
            elif H == 1 and W > 1:
                t2 = go.sum(dim=2)
                _replication_pad2d_backward_colfold_kernel[
                    (triton.cdiv(NC * 1 * W, 256),)
                ](
                    t2.reshape(NC, 1, OW),
                    gi,
                    OW,
                    W,
                    pl,
                    pr,
                    1,
                    NC * 1 * W,
                    MAXG=max(pl + 1, pr + 1, 1),
                    BLOCK=256,
                )
            else:
                gi.fill_(go.sum())
        elif pl < 0 or pr < 0 or pt < 0 or pb < 0:
            maxg_col = max(pl + 1, pr + 1, 1)
            maxg_row = max(pt + 1, pb + 1, 1)
            cf = torch.empty(NC, OH, W, device=x.device, dtype=torch.float32)
            _replication_pad2d_backward_colfold_kernel[
                (triton.cdiv(NC * OH * W, 256),)
            ](
                go,
                cf,
                OW,
                W,
                pl,
                pr,
                OH,
                NC * OH * W,
                MAXG=maxg_col,
                BLOCK=256,
            )
            _replication_pad2d_backward_rowfold_kernel[(triton.cdiv(NC * H * W, 256),)](
                cf,
                gi,
                W,
                H,
                pt,
                pb,
                OH,
                NC * H * W,
                MAXG=maxg_row,
                BLOCK=256,
            )
        else:
            maxg = max(pt + 1, pb + 1, pl + 1, pr + 1, 1)
            BLOCK = 1024
            h2 = H - 2
            if W <= 512:
                _replication_pad2d_backward_bulk_contig_kernel[
                    (NC, triton.cdiv(h2 * W, BLOCK))
                ](
                    go,
                    gi,
                    OW,
                    W,
                    h2,
                    pt,
                    pl,
                    OHW,
                    HW,
                    h2 * W,
                    BLOCK=BLOCK,
                )
            else:
                need_mask = (W % BLOCK) != 0
                cpw = triton.cdiv(W, BLOCK)
                _replication_pad2d_backward_bulk_kernel[(NC * h2 * cpw,)](
                    go,
                    gi,
                    OW,
                    W,
                    h2,
                    pt,
                    pl,
                    OHW,
                    HW,
                    CPW=cpw,
                    NEED_MASK=need_mask,
                    BLOCK=BLOCK,
                )
            cpw_e = triton.cdiv(W, BLOCK)
            _replication_pad2d_backward_row_edge_kernel[(NC, 2, cpw_e)](
                go,
                gi,
                OW,
                W,
                H,
                pt,
                pl,
                pr,
                pb,
                OHW,
                HW,
                MAXG=maxg,
                BLOCK=BLOCK,
            )
            n_col = NC * 2 * h2
            if n_col > 0:
                P = 1
                while P < h2:
                    P *= 2
                P = min(max(P, 1), 1024)
                _replication_pad2d_backward_col_edge_kernel[
                    (NC, 2, triton.cdiv(h2, P))
                ](
                    go,
                    gi,
                    OW,
                    W,
                    h2,
                    pt,
                    pl,
                    pr,
                    OHW,
                    HW,
                    MAXG=maxg,
                    P=P,
                )

    res = gi.view(N, C, H, W)

    if out is not None:
        torch.ops.aten._copy_from(res, out)
        return out
    return res.view(self.shape) if is_3d else res


def replication_pad2d_backward(
    grad_output: torch.Tensor, self: torch.Tensor, padding
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(
    grad_output: torch.Tensor,
    self: torch.Tensor,
    padding,
    *,
    grad_input: torch.Tensor,
) -> torch.Tensor:
    logger.debug("GEMS_KUNLUNXIN REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
