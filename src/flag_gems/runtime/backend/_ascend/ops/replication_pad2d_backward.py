import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import triton_lang_extension as ext

logger = logging.getLogger(__name__)

# rows owned by one program.  Only the accumulator row-vectors are live at a
# time, so this does NOT cost UB: every buffer is 1xWN regardless of R.
DEFAULT_R = 16


@triton.jit
def _backward_rows_kernel(
    gop,
    gip,
    CO,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    R: tl.constexpr,
    WN: tl.constexpr,
    FULL: tl.constexpr,
    NEED_TOP: tl.constexpr,
    NEED_BOT: tl.constexpr,
    H1: tl.constexpr,
):
    pid = ext.program_id(0)
    n_row_blocks = (H + R - 1) // R
    nc = CO + pid // n_row_blocks
    j0 = (pid % n_row_blocks) * R
    c_out = nc * OH * OW
    c_in = nc * H * W
    iw = tl.arange(0, WN)
    cv = iw < W

    for k in tl.range(0, R):
        j = j0 + k
        if j < H:
            # ---- aligned interior value for input row j (output row j+PT).
            # When W is already the padded arange width (FULL) every lane is
            # valid and the transfer is a plain bulk DMA - much faster on wide
            # rows than a masked (lane-selected) transfer.
            if FULL:
                row = tl.load(gop + c_out + (j + PT) * OW + (PL + iw))
            else:
                row = tl.load(
                    gop + c_out + (j + PT) * OW + (PL + iw),
                    mask=cv,
                    other=0.0,
                )
            # ---- horizontal pad columns of this aligned row fold into the
            # two edge lanes of the same input row.
            if PL > 0:
                left = tl.sum(
                    tl.load(gop + c_out + (j + PT) * OW + tl.arange(0, PL)),
                    axis=0,
                )
                row = tl.where(iw == 0, row + left, row)
            if PR > 0:
                right = tl.sum(
                    tl.load(gop + c_out + (j + PT) * OW + (PL + W + tl.arange(0, PR))),
                    axis=0,
                )
                row = tl.where(iw == W - 1, row + right, row)

            # ---- top pad rows fold onto input row 0.
            # (NEED_TOP is constexpr but the & with the runtime (j == 0) makes
            # this a device branch; every arange inside is still guarded by a
            # standalone constexpr `if X > 0` so degenerate pads never trace
            # an empty arange.)
            if (NEED_TOP) & (j == 0):
                if PT > 0:
                    # whole pad rows -> row 0
                    for t in tl.static_range(0, PT):
                        if FULL:
                            top_row = tl.load(gop + c_out + t * OW + (PL + iw))
                        else:
                            top_row = tl.load(
                                gop + c_out + t * OW + (PL + iw),
                                mask=cv,
                                other=0.0,
                            )
                        row += top_row
                    # their pad columns -> the two corner lanes of row 0
                    if PL > 0:
                        lc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + tl.arange(0, PT)[:, None] * OW
                                + tl.arange(0, PL)[None, :]
                            )
                        )
                        row = tl.where(iw == 0, row + lc, row)
                    if PR > 0:
                        rc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + tl.arange(0, PT)[:, None] * OW
                                + (PL + W + tl.arange(0, PR))[None, :]
                            )
                        )
                        row = tl.where(iw == W - 1, row + rc, row)

            # ---- when H == 1 the bottom pad rows also fold onto row 0.
            if (H1 == 1) & (j == 0):
                if PB > 0:
                    for b in tl.static_range(0, PB):
                        if FULL:
                            bot_row = tl.load(
                                gop + c_out + (PT + H + b) * OW + (PL + iw)
                            )
                        else:
                            bot_row = tl.load(
                                gop + c_out + (PT + H + b) * OW + (PL + iw),
                                mask=cv,
                                other=0.0,
                            )
                        row += bot_row
                    if PL > 0:
                        lc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + tl.arange(0, PL)[None, :]
                            )
                        )
                        row = tl.where(iw == 0, row + lc, row)
                    if PR > 0:
                        rc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + (PL + W + tl.arange(0, PR))[None, :]
                            )
                        )
                        row = tl.where(iw == W - 1, row + rc, row)

            # ---- bottom pad rows fold onto row H-1 (H > 1 here).
            if (NEED_BOT) & (j == H - 1):
                if PB > 0:
                    for b in tl.static_range(0, PB):
                        if FULL:
                            bot_row = tl.load(
                                gop + c_out + (PT + H + b) * OW + (PL + iw)
                            )
                        else:
                            bot_row = tl.load(
                                gop + c_out + (PT + H + b) * OW + (PL + iw),
                                mask=cv,
                                other=0.0,
                            )
                        row += bot_row
                    if PL > 0:
                        lc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + tl.arange(0, PL)[None, :]
                            )
                        )
                        row = tl.where(iw == 0, row + lc, row)
                    if PR > 0:
                        rc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + (PL + W + tl.arange(0, PR))[None, :]
                            )
                        )
                        row = tl.where(iw == W - 1, row + rc, row)

            if FULL:
                tl.store(gip + c_in + j * W + iw, row)
            else:
                tl.store(gip + c_in + j * W + iw, row, mask=cv)


# ---------------------------------------------------------------------------
# Wide fast path (large tensors with power-of-two W).
#
# The row loop above is correct everywhere but is bandwidth-starved on big
# tensors: each program walks its R rows one 1xWN vector at a time, so DMA
# never saturates (~250 GB/s vs torch's ~550 GB/s).  For wide inputs we split
# the op into two ordered launches that each keep small UB footprints:
#
#   * K1 broadcasts the aligned value of every input row as one R x WN 2D tile
#     (a plain bulk DMA), folding the horizontal pad columns of each aligned
#     row onto that tile's two edge lanes.  Rows 0 and H-1 are also written
#     here but are *overwritten* below.
#   * K2 (launched after K1 on the same stream) rewrites input rows 0 and H-1
#     from scratch as the full replication sum of every output row that maps
#     onto them (top/bottom pad rows + the aligned row), including corners.
#
# K1 and K2 target disjoint or ordered cells, so every input element is decided
# by exactly one launch and the result is race-free.
# ---------------------------------------------------------------------------
@triton.jit
def _backward_wide_k1(
    gop,
    gip,
    CO,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    R: tl.constexpr,
    WN: tl.constexpr,
):
    pid = ext.program_id(0)
    nrb = (H + R - 1) // R
    nc = CO + pid // nrb
    j0 = (pid % nrb) * R
    c_out = nc * OH * OW
    c_in = nc * H * W
    rr = j0 + tl.arange(0, R)
    iw = tl.arange(0, WN)
    m = rr < H

    A = tl.load(
        gop + c_out + (rr + PT)[:, None] * OW + (PL + iw)[None, :],
        mask=m[:, None],
        other=0.0,
    )
    # Horizontal pad columns of the aligned row fold onto its two edge lanes.
    # A width of one also goes through a 2-wide tile with the second lane masked
    # off: the plain strided R-vector this replaces lowers ~100x slower on bf16,
    # while a degenerate [R x 1] tile lowers ~100x slower on fp16, so neither
    # width can use its "natural" form.  The masked second lane is never read.
    if PL == 1:
        one = tl.arange(0, 2)[None, :]
        L = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + one,
                mask=m[:, None] & (one == 0),
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == 0, A + L[:, None], A)
    elif PL > 1:
        L = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + tl.arange(0, PL)[None, :],
                mask=m[:, None],
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == 0, A + L[:, None], A)
    if PR == 1:
        one = tl.arange(0, 2)[None, :]
        Rv = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + (PL + W + one),
                mask=m[:, None] & (one == 0),
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == W - 1, A + Rv[:, None], A)
    elif PR > 1:
        Rv = tl.sum(
            tl.load(
                gop
                + c_out
                + (rr + PT)[:, None] * OW
                + (PL + W + tl.arange(0, PR))[None, :],
                mask=m[:, None],
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == W - 1, A + Rv[:, None], A)
    tl.store(gip + c_in + rr[:, None] * W + iw[None, :], A, mask=m[:, None])


@triton.jit
def _backward_wide_k2(
    gop,
    gip,
    CO,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    WN: tl.constexpr,
    NEED0: tl.constexpr,
    NEEDH: tl.constexpr,
    CN0: tl.constexpr,
    CNH: tl.constexpr,
):
    pid = ext.program_id(0)
    nc = CO + pid
    c_out = nc * OH * OW
    c_in = nc * H * W
    iw = tl.arange(0, WN)

    # input row 0 <- top pad rows r in [0, PT) plus the aligned row r == PT
    # (CN0 rows starting at r == 0).  Rows 0 and H-1 only ever map onto
    # themselves as the aligned copy, which K1 already wrote; here we rewrite
    # the whole row with every contribution, so pad sums are counted once.
    if NEED0:
        acc = tl.sum(
            tl.load(gop + c_out + tl.arange(0, CN0)[:, None] * OW + (PL + iw)[None, :]),
            axis=0,
        )
        if PL == 1:
            lc = tl.sum(tl.load(gop + c_out + tl.arange(0, CN0) * OW + 0))
            acc = tl.where(iw == 0, acc + lc, acc)
        elif PL > 1:
            lc = tl.sum(
                tl.load(
                    gop
                    + c_out
                    + tl.arange(0, CN0)[:, None] * OW
                    + tl.arange(0, PL)[None, :]
                )
            )
            acc = tl.where(iw == 0, acc + lc, acc)
        if PR == 1:
            rc = tl.sum(tl.load(gop + c_out + tl.arange(0, CN0) * OW + (PL + W)))
            acc = tl.where(iw == W - 1, acc + rc, acc)
        elif PR > 1:
            rc = tl.sum(
                tl.load(
                    gop
                    + c_out
                    + tl.arange(0, CN0)[:, None] * OW
                    + (PL + W + tl.arange(0, PR))[None, :]
                )
            )
            acc = tl.where(iw == W - 1, acc + rc, acc)
        tl.store(gip + c_in + iw, acc)

    # input row H-1 <- bottom pad rows plus the aligned last row r == PT+H-1.
    if NEEDH:
        rb = PT + H - 1
        acc = tl.sum(
            tl.load(
                gop
                + c_out
                + (rb + tl.arange(0, CNH))[:, None] * OW
                + (PL + iw)[None, :]
            ),
            axis=0,
        )
        if PL == 1:
            lc = tl.sum(tl.load(gop + c_out + (rb + tl.arange(0, CNH)) * OW + 0))
            acc = tl.where(iw == 0, acc + lc, acc)
        elif PL > 1:
            lc = tl.sum(
                tl.load(
                    gop
                    + c_out
                    + (rb + tl.arange(0, CNH))[:, None] * OW
                    + tl.arange(0, PL)[None, :]
                )
            )
            acc = tl.where(iw == 0, acc + lc, acc)
        if PR == 1:
            rc = tl.sum(tl.load(gop + c_out + (rb + tl.arange(0, CNH)) * OW + (PL + W)))
            acc = tl.where(iw == W - 1, acc + rc, acc)
        elif PR > 1:
            rc = tl.sum(
                tl.load(
                    gop
                    + c_out
                    + (rb + tl.arange(0, CNH))[:, None] * OW
                    + (PL + W + tl.arange(0, PR))[None, :]
                )
            )
            acc = tl.where(iw == W - 1, acc + rc, acc)
        tl.store(gip + c_in + (H - 1) * W + iw, acc)


# ---------------------------------------------------------------------------
# Single-launch wide path (medium tensors with power-of-two W).
#
# K1+K2 above beat the row loop by 5-7x once DMA dominates, but the second
# launch sits right at the launcher floor (~50us on this part), which is pure
# loss for tensors that are too small to be bandwidth-bound.  This kernel folds
# K2's job into K1: the block that owns input row 0 / H-1 rewrites those rows
# with every vertical contribution before storing the tile, so the whole op is
# one launch.  Rows 0 and H-1 are only ever the aligned copy of themselves, so
# folding in-place counts each pad row exactly once.
#
# The cost is headroom: the extra folds roughly halve the tile that fits in UB,
# so R stays <= M_R here.  That is fine in this size band (the grid is at the
# launcher floor either way) but would starve the large shapes, which keep
# using the two-launch path above.
# ---------------------------------------------------------------------------
@triton.jit
def _backward_wide_merged(
    gop,
    gip,
    CO,
    OH,
    OW,
    H,
    W,
    PL: tl.constexpr,
    PR: tl.constexpr,
    PT: tl.constexpr,
    PB: tl.constexpr,
    R: tl.constexpr,
    WN: tl.constexpr,
    NEED0: tl.constexpr,
    NEEDH: tl.constexpr,
    H1: tl.constexpr,
):
    pid = ext.program_id(0)
    nrb = (H + R - 1) // R
    nc = CO + pid // nrb
    j0 = (pid % nrb) * R
    c_out = nc * OH * OW
    c_in = nc * H * W
    rr = j0 + tl.arange(0, R)
    iw = tl.arange(0, WN)
    m = rr < H

    A = tl.load(
        gop + c_out + (rr + PT)[:, None] * OW + (PL + iw)[None, :],
        mask=m[:, None],
        other=0.0,
    )
    # Horizontal pad columns of the aligned row fold onto its two edge lanes.
    # A width of one also goes through a 2-wide tile with the second lane masked
    # off: the plain strided R-vector this replaces lowers ~100x slower on bf16,
    # while a degenerate [R x 1] tile lowers ~100x slower on fp16, so neither
    # width can use its "natural" form.  The masked second lane is never read.
    if PL == 1:
        one = tl.arange(0, 2)[None, :]
        L = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + one,
                mask=m[:, None] & (one == 0),
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == 0, A + L[:, None], A)
    elif PL > 1:
        L = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + tl.arange(0, PL)[None, :],
                mask=m[:, None],
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == 0, A + L[:, None], A)
    if PR == 1:
        one = tl.arange(0, 2)[None, :]
        Rv = tl.sum(
            tl.load(
                gop + c_out + (rr + PT)[:, None] * OW + (PL + W + one),
                mask=m[:, None] & (one == 0),
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == W - 1, A + Rv[:, None], A)
    elif PR > 1:
        Rv = tl.sum(
            tl.load(
                gop
                + c_out
                + (rr + PT)[:, None] * OW
                + (PL + W + tl.arange(0, PR))[None, :],
                mask=m[:, None],
                other=0.0,
            ),
            axis=1,
        )
        A = tl.where(iw[None, :] == W - 1, A + Rv[:, None], A)

    # ---- top pad rows fold onto input row 0, handled by its owning block.
    # The pad rows are walked one 1xWN vector at a time rather than loaded as a
    # [PT x WN] tile: the 2D form needs an extra live buffer big enough to push
    # the whole kernel over UB.
    if NEED0:
        if j0 == 0:
            if PT > 0:
                for t in tl.static_range(0, PT):
                    top_row = tl.load(gop + c_out + t * OW + (PL + iw))
                    A = A + tl.where(rr[:, None] == 0, top_row[None, :], 0.0)
                if PL > 0:
                    lc = tl.sum(
                        tl.load(
                            gop
                            + c_out
                            + tl.arange(0, PT)[:, None] * OW
                            + tl.arange(0, PL)[None, :]
                        )
                    )
                    A = tl.where((rr[:, None] == 0) & (iw[None, :] == 0), A + lc, A)
                if PR > 0:
                    rc = tl.sum(
                        tl.load(
                            gop
                            + c_out
                            + tl.arange(0, PT)[:, None] * OW
                            + (PL + W + tl.arange(0, PR))[None, :]
                        )
                    )
                    A = tl.where((rr[:, None] == 0) & (iw[None, :] == W - 1), A + rc, A)
            # when H == 1 the bottom pad rows land on row 0 as well
            if H1:
                if PB > 0:
                    for b in tl.static_range(0, PB):
                        bot_row = tl.load(gop + c_out + (PT + H + b) * OW + (PL + iw))
                        A = A + tl.where(rr[:, None] == 0, bot_row[None, :], 0.0)
                    if PL > 0:
                        lc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + tl.arange(0, PL)[None, :]
                            )
                        )
                        A = tl.where((rr[:, None] == 0) & (iw[None, :] == 0), A + lc, A)
                    if PR > 0:
                        rc = tl.sum(
                            tl.load(
                                gop
                                + c_out
                                + (PT + H + tl.arange(0, PB))[:, None] * OW
                                + (PL + W + tl.arange(0, PR))[None, :]
                            )
                        )
                        A = tl.where(
                            (rr[:, None] == 0) & (iw[None, :] == W - 1), A + rc, A
                        )
    # ---- bottom pad rows fold onto input row H-1 (H > 1 here).
    if NEEDH:
        if j0 + R >= H:
            if PB > 0:
                for b in tl.static_range(0, PB):
                    bot_row = tl.load(gop + c_out + (PT + H + b) * OW + (PL + iw))
                    A = A + tl.where(rr[:, None] == H - 1, bot_row[None, :], 0.0)
                if PL > 0:
                    lc = tl.sum(
                        tl.load(
                            gop
                            + c_out
                            + (PT + H + tl.arange(0, PB))[:, None] * OW
                            + tl.arange(0, PL)[None, :]
                        )
                    )
                    A = tl.where((rr[:, None] == H - 1) & (iw[None, :] == 0), A + lc, A)
                if PR > 0:
                    rc = tl.sum(
                        tl.load(
                            gop
                            + c_out
                            + (PT + H + tl.arange(0, PB))[:, None] * OW
                            + (PL + W + tl.arange(0, PR))[None, :]
                        )
                    )
                    A = tl.where(
                        (rr[:, None] == H - 1) & (iw[None, :] == W - 1), A + rc, A
                    )

    tl.store(gip + c_in + rr[:, None] * W + iw[None, :], A, mask=m[:, None])


# element budget of the merged [R x W] tile; half the two-launch budget because
# the in-kernel folds keep several more [R x W] temporaries live.
MERGED_TILE_ELEMS = 4096
MERGED_MAX_R = 32


def _has_edges(pl, pr, pt, pb):
    return pl > 0 or pr > 0 or pt > 0 or pb > 0


_plan_cache = {}


def _replication_pad2d_backward_impl(
    grad_output: torch.Tensor, self: torch.Tensor, padding, *, out: torch.Tensor = None
) -> torch.Tensor:
    if isinstance(padding, torch.Tensor):
        padding = tuple(padding.tolist())
    if isinstance(padding, int):
        pad_left = pad_right = pad_top = pad_bottom = padding
    elif isinstance(padding, (tuple, list)):
        if len(padding) != 4:
            raise ValueError(
                "padding must be a sequence of 4 integers: "
                "(pad_left, pad_right, pad_top, pad_bottom)"
            )
        pad_left, pad_right, pad_top, pad_bottom = map(int, padding)
    else:
        raise TypeError(f"Unexpected padding type: {type(padding)}")

    if pad_left < 0 or pad_right < 0 or pad_top < 0 or pad_bottom < 0:
        raise ValueError("Padding values must be non-negative")

    is_3d = self.ndim == 3
    if is_3d:
        C, H, W = self.shape
        n_images = C
    elif self.ndim == 4:
        C, H, W = self.shape[1], self.shape[2], self.shape[3]
        n_images = self.shape[0] * C
    else:
        raise ValueError("replication_pad2d_backward expects 3D or 4D input")

    # The kernel addresses both tensors with purely linear offsets
    # (image_base = nc * OH * OW / nc * H * W); contiguous 3D/4D tensors share
    # that layout, so no view/reshape is needed in the hot path.
    if not grad_output.is_contiguous():
        grad_output = grad_output.contiguous()
    if out is not None and not out.is_contiguous():
        out = out.contiguous()

    OH = H + pad_top + pad_bottom
    OW = W + pad_left + pad_right

    if not _has_edges(pad_left, pad_right, pad_top, pad_bottom):
        result = grad_output.to(self.dtype)
        if out is not None:
            out.copy_(result)
            return out
        if is_3d:
            return result.view(C, H, W)
        return result

    device = self.device
    if out is None:
        out = torch.empty(self.shape, device=device, dtype=self.dtype)

    itemsize = grad_output.element_size()

    # The launch plan depends only on the input geometry -- W decides whether a
    # row is a plain DMA block, H decides how many rows one program owns -- so
    # the cache key is just (H, W) and the hot path pays one 2-tuple hash.
    plan = _plan_cache.get((H, W))
    if plan is None:
        # WIDTH: the widest row is a plain DMA block when W equals its padded
        # arange width (W a power of two); otherwise lanes past W are masked.
        if (W & (W - 1)) == 0:
            w_width = W
            full = True
        else:
            w_width = triton.next_power_of_2(W)
            full = False
        rows_per_block = max(1, min(H, max(DEFAULT_R, (H + 255) // 256)))
        plan = (w_width, full, rows_per_block)
        _plan_cache[(H, W)] = plan
    w_width, full, rows_per_block = plan

    # 2D-tile paths: only for the shapes they are designed and tested for -- a
    # power-of-two W so the bulk transfer is unmasked, non-degenerate H, and
    # small pads so every tile stays well inside the on-chip buffer.
    if (
        full
        and W >= 32
        and H >= 8
        and pad_left <= 8
        and pad_right <= 8
        and pad_top <= 8
        and pad_bottom <= 8
    ):
        need0 = bool(pad_top > 0 or (H == 1 and pad_bottom > 0))
        needh = bool(pad_bottom > 0 and H > 1)
        h1 = bool(H == 1)
        nbytes = self.numel() * itemsize

        # Which 2D path: the merged kernel is a single launch, which saves a
        # whole launcher round-trip, but the folds cap its tile at MERGED_MAX_R
        # rows while the split can put four times more rows in UB.  Measured
        # crossover: merged wins up to ~16MB / 32K rows of input and loses once
        # the transfer is large enough for the bigger tile to pay for itself.
        if (
            nbytes >= 1024 * 1024
            and nbytes < 24 * 1024 * 1024
            and n_images * H <= 32768
        ):
            # Medium: same 2D transfer folded into a single launch, since a
            # second launch costs a launcher round-trip (~50us) that outweighs
            # any larger tile this size could use; see _backward_wide_merged.
            r_merged = max(
                1, min(H, MERGED_MAX_R, (MERGED_TILE_ELEMS * 2 // itemsize) // W)
            )
            grid = (n_images * triton.cdiv(H, r_merged),)
            _backward_wide_merged[grid](
                grad_output,
                out,
                0,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                r_merged,
                w_width,
                need0,
                needh,
                h1,
            )
            return out

        if nbytes >= 8 * 1024 * 1024:
            # Large: splitting the op in two lets both launches use the biggest
            # 2D tile UB allows, which is what saturates DMA here.
            #
            # R * WN tiles keep the 2D copies far below the UB cap.  The budget
            # is in BYTES, and 16384 elements was tuned for 2-byte dtypes, so
            # scale the element cap by element size: fp32 must use half the rows
            # of fp16/bf16 or the [R x W] tile overflows the 192KB UB and fails
            # to compile.
            r_wide = max(1, min(H, (16384 * 2 // itemsize) // W))
            grid = (n_images * triton.cdiv(H, r_wide),)
            _backward_wide_k1[grid](
                grad_output,
                out,
                0,
                OH,
                OW,
                H,
                W,
                pad_left,
                pad_right,
                pad_top,
                r_wide,
                w_width,
            )
            # K1 already seeded rows 0/H-1 with the aligned value; K2 rewrites
            # them with every vertical contribution.  Skip it entirely when
            # there are no vertical pads (K1 alone is then exact).
            if need0 or needh:
                cn0 = pad_top + 1 + (pad_bottom if h1 else 0)
                cnh = pad_bottom + 1
                _backward_wide_k2[(n_images,)](
                    grad_output,
                    out,
                    0,
                    OH,
                    OW,
                    H,
                    W,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    w_width,
                    need0,
                    needh,
                    cn0,
                    cnh,
                )
            return out

    need_top = bool(pad_top > 0 or (H == 1 and pad_bottom > 0))
    need_bot = bool(pad_bottom > 0 and H > 1)
    h1 = bool(H == 1)
    grid = (n_images * triton.cdiv(H, rows_per_block),)
    _backward_rows_kernel[grid](
        grad_output,
        out,
        0,
        OH,
        OW,
        H,
        W,
        pad_left,
        pad_right,
        pad_top,
        pad_bottom,
        rows_per_block,
        w_width,
        full,
        need_top,
        need_bot,
        h1,
    )
    return out


def replication_pad2d_backward(grad_output, self, padding):
    logger.debug("GEMS_ASCEND REPLICATION_PAD2D_BACKWARD")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=None)


def replication_pad2d_backward_grad_input(grad_output, self, padding, *, grad_input):
    logger.debug("GEMS_ASCEND REPLICATION_PAD2D_BACKWARD_GRAD_INPUT")
    return _replication_pad2d_backward_impl(grad_output, self, padding, out=grad_input)
