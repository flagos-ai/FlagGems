import torch
import triton
import triton.language as tl


@triton.jit
def _addmv_kernel(
    mat_ptr,
    vec_ptr,
    partial_ptr,
    counter_ptr,
    self_ptr,
    M,
    N,
    stride_m,
    stride_k,
    stride_v,
    stride_self,
    beta,
    alpha,
    nrb,
    K_SPLITS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    ACC_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m = pid // K_SPLITS
    pid_k = pid % K_SPLITS

    lane_off = tl.arange(0, BM)
    row_off = pid_m * BM + lane_off
    row_mask = row_off < M

    k_start = pid_k * BN
    k_lim = k_start + BN

    acc = tl.zeros((BM,), dtype=ACC_DTYPE)
    for k0 in range(0, BN, BK):
        k_off = k_start + k0 + tl.arange(0, BK)
        km = (k_off < k_lim) & (k_off < N)
        m_tile = tl.load(
            mat_ptr + row_off[:, None] * stride_m + k_off[None, :] * stride_k,
            mask=row_mask[:, None] & km[None, :],
            other=0.0,
        )
        v_tile = tl.load(vec_ptr + k_off * stride_v, mask=km, other=0.0)
        acc += tl.sum(m_tile.to(ACC_DTYPE) * v_tile.to(ACC_DTYPE)[None, :], axis=1)

    if K_SPLITS == 1:
        self_tile = tl.load(self_ptr + row_off * stride_self, mask=row_mask, other=0.0)
        res = beta * self_tile.to(ACC_DTYPE) + alpha * acc
        tl.store(
            self_ptr + row_off * stride_self,
            res.to(self_ptr.dtype.element_ty),
            mask=row_mask,
        )
    else:
        tl.store(
            partial_ptr + pid_k * nrb * BM + pid_m * BM + lane_off,
            acc,
            mask=row_mask,
        )
        # per-row-block arrival counter: the last program of each row block
        # reduces that block's K partials and applies beta/alpha in parallel
        old = tl.atomic_add(counter_ptr + pid_m, 1, sem="acq_rel")
        if old == K_SPLITS - 1:
            s = tl.zeros((BM,), dtype=ACC_DTYPE)
            for k in range(0, K_SPLITS):
                p = tl.load(
                    partial_ptr + k * nrb * BM + pid_m * BM + lane_off,
                    mask=row_mask,
                    other=0.0,
                )
                s += p
            self_tile = tl.load(
                self_ptr + row_off * stride_self, mask=row_mask, other=0.0
            )
            res = beta * self_tile.to(ACC_DTYPE) + alpha * s
            tl.store(
                self_ptr + row_off * stride_self,
                res.to(self_ptr.dtype.element_ty),
                mask=row_mask,
            )


def run(self, mat, vec, *, beta=1, alpha=1):
    if isinstance(beta, torch.Tensor):
        beta = beta.item()
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    beta = float(beta)
    alpha = float(alpha)

    M, N = mat.shape
    if M == 0 or N == 0:
        return self

    BK = 128

    if N <= 1024:
        K_splits = 1
        BM = 16
        nw = 4
    elif N <= 4 * M:
        # square-ish: single-pass per row block, no split finalize needed
        K_splits = 1
        BM = 32
        nw = 4
    else:
        if self.dtype == torch.float32 and N > 8192:
            BM = 32
            maxk = 32
        else:
            BM = 64
            maxk = 8
        nw = 4
        K_splits = min((N + 511) // 512, maxk)

    nrb = (M + BM - 1) // BM

    is_fp64 = self.dtype == torch.float64
    acc_dtype = torch.float64 if is_fp64 else torch.float32
    tl_acc = tl.float64 if is_fp64 else tl.float32

    BN = (N + K_splits - 1) // K_splits
    grid = (nrb * K_splits,)

    if K_splits == 1:
        _addmv_kernel[grid](
            mat,
            vec,
            self,
            self,
            self,
            M,
            N,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
            beta,
            alpha,
            nrb,
            K_SPLITS=1,
            BM=BM,
            BK=BK,
            BN=BN,
            ACC_DTYPE=tl_acc,
            num_warps=nw,
            num_stages=3,
        )
    else:
        partial = torch.empty((K_splits, nrb, BM), dtype=acc_dtype, device=self.device)
        counter = torch.zeros(nrb, dtype=torch.int32, device=self.device)
        _addmv_kernel[grid](
            mat,
            vec,
            partial,
            counter,
            self,
            M,
            N,
            mat.stride(0),
            mat.stride(1),
            vec.stride(0),
            self.stride(0),
            beta,
            alpha,
            nrb,
            K_SPLITS=K_splits,
            BM=BM,
            BK=BK,
            BN=BN,
            ACC_DTYPE=tl_acc,
            num_warps=nw,
            num_stages=3,
        )
    return self


# Alias for FlagGems import convention
addmv_ = run
