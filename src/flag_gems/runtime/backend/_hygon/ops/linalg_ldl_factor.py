import torch
import triton
import triton.language as tl


@triton.jit
def _ldl_factor_kernel(
    A,
    LD,
    PIV,
    n,
    stride_ab,
    stride_am,
    stride_an,
    stride_lb,
    stride_lm,
    stride_ln,
    stride_pb,
    BLOCK_N: tl.constexpr,
    NB: tl.constexpr,
    KC: tl.constexpr,
):
    # One program per matrix in the batch.  Blocked left-looking LDL^T
    # (no pivoting), matching the flag_gems reference semantics:
    #   * input symmetrized in-kernel: LD = (A + A^T)/2
    #   * D on the diagonal, L (unit-diagonal implicit) below, zeros above
    #   * pivots = identity (1-based, int32)
    bid = tl.program_id(0)
    a_base = A + bid * stride_ab
    l_base = LD + bid * stride_lb
    p_base = PIV + bid * stride_pb

    rows = tl.arange(0, BLOCK_N)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < n
    cmask = cols < n

    # LD = (A + A^T) / 2
    a_ij = tl.load(
        a_base + rows[:, None] * stride_am + cols[None, :] * stride_an,
        mask=rmask[:, None] & cmask[None, :],
        other=0.0,
    )
    a_ji = tl.load(
        a_base + cols[None, :] * stride_am + rows[:, None] * stride_an,
        mask=rmask[:, None] & cmask[None, :],
        other=0.0,
    )
    tl.store(
        l_base + rows[:, None] * stride_lm + cols[None, :] * stride_ln,
        (a_ij + a_ji) * 0.5,
        mask=rmask[:, None] & cmask[None, :],
    )

    nb_cols = tl.arange(0, NB)
    rowidx = tl.arange(0, BLOCK_N)
    colidx = tl.arange(0, NB)

    for kb in tl.range(0, n, NB, num_stages=2):
        kb_end = kb + NB
        if kb_end > n:
            kb_end = n
        p_rows = kb + rows
        p_mask = p_rows < n
        pcols = kb + nb_cols

        # Load the raw panel (rows kb..n, cols kb..kb_end) into registers.
        panel = tl.load(
            l_base + p_rows[:, None] * stride_lm + pcols[None, :] * stride_ln,
            mask=p_mask[:, None] & (pcols[None, :] < kb_end),
            other=0.0,
        )

        # Left-looking panel update: subtract sum_{j<kb} L[:,j]*D[j]*L[block,j]^T
        jc = 0
        while jc < kb:
            j = jc + tl.arange(0, KC)
            jmask = j < kb
            tile1 = tl.load(
                l_base + p_rows[:, None] * stride_lm + j[None, :] * stride_ln,
                mask=p_mask[:, None] & jmask[None, :],
                other=0.0,
            )
            tile2 = tl.load(
                l_base + pcols[:, None] * stride_lm + j[None, :] * stride_ln,
                mask=(pcols[:, None] < kb_end) & jmask[None, :],
                other=0.0,
            )
            d_j = tl.load(l_base + j * stride_lm + j * stride_ln, mask=jmask, other=0.0)
            panel -= tl.dot(
                tile1, tl.trans(tile2 * d_j[None, :]), input_precision="ieee"
            )
            jc += KC

        # Block-diagonal D values (first NB entries).
        d_block = tl.sum(
            tl.where(rowidx[:, None] == colidx[None, :], panel, 0.0), axis=0
        )

        # Unblocked factor of the diagonal block on the register panel:
        #   D[kk]   = S[kk,kk] - sum_{j<kk} L[kk,j]^2 * D[j]
        #   L[i,kk] = (S[i,kk] - sum_{j<kk} L[i,j]*L[kk,j]*D[j]) / D[kk]
        for kk in range(kb, kb_end):
            c = kk - kb
            row_kk = tl.sum(tl.where(rowidx[:, None] == c, panel, 0.0), axis=0)
            w = tl.where(colidx < c, row_kk * d_block, 0.0)
            acc_i = tl.sum(panel * w[None, :], axis=1)
            raw_col = tl.sum(tl.where(colidx[None, :] == c, panel, 0.0), axis=1)
            s_kk = tl.sum(tl.where(colidx == c, row_kk, 0.0), axis=0)
            dsum = tl.sum(
                tl.where(colidx < c, (row_kk * row_kk) * d_block, 0.0), axis=0
            )
            d_kk = s_kk - dsum
            new_col = (raw_col - acc_i) / d_kk
            panel = tl.where(colidx[None, :] == c, new_col[:, None], panel)
            panel = tl.where(
                (rowidx[:, None] == c) & (colidx[None, :] == c), d_kk, panel
            )
            d_block = tl.where(colidx == c, d_kk, d_block)

        # Store the factored panel.
        tl.store(
            l_base + p_rows[:, None] * stride_lm + pcols[None, :] * stride_ln,
            panel,
            mask=p_mask[:, None] & (pcols[None, :] < kb_end),
        )

    # Zero the upper triangle (i < j).
    tl.store(
        l_base + rows[:, None] * stride_lm + cols[None, :] * stride_ln,
        0.0,
        mask=(rows[:, None] < cols[None, :]) & rmask[:, None] & cmask[None, :],
    )

    # Pivots are the identity permutation (1-based), like the reference.
    tl.store(p_base + rows, rows + 1, mask=rmask)


def run(self, *, hermitian=False):
    if self.ndim < 2:
        raise ValueError("linalg_ldl_factor: A must be at least 2D")
    if self.shape[-2] != self.shape[-1]:
        raise ValueError("linalg_ldl_factor: matrix must be square")
    if self.dtype not in (torch.float32, torch.float64):
        raise TypeError("linalg_ldl_factor: only float32 and float64 are supported")

    batch_shape = self.shape[:-2]
    n = self.shape[-1]
    batch = 1
    for d in batch_shape:
        batch *= d

    if n > 256:
        raise ValueError("linalg_ldl_factor: matrix size exceeds supported maximum")

    LD = torch.empty(self.shape, dtype=self.dtype, device=self.device)
    pivots = torch.empty(batch_shape + (n,), dtype=torch.int32, device=self.device)

    BLOCK_N = triton.next_power_of_2(n)
    NB = 16 if n > 16 else max(BLOCK_N, 1)
    if NB > BLOCK_N:
        NB = BLOCK_N
    KC = 16
    grid = (batch,)
    _ldl_factor_kernel[grid](
        self,
        LD,
        pivots,
        n,
        self.stride(0) if self.ndim > 2 else 0,
        self.stride(-2),
        self.stride(-1),
        LD.stride(0) if LD.ndim > 2 else 0,
        LD.stride(-2),
        LD.stride(-1),
        pivots.stride(0) if pivots.ndim > 1 else 0,
        BLOCK_N=BLOCK_N,
        NB=NB,
        KC=KC,
    )
    return (LD, pivots)


# Alias for FlagGems import convention
linalg_ldl_factor = run
