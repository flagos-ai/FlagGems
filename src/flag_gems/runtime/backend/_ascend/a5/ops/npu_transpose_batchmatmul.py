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

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn


@triton.jit
def _load_ab(
    a_ptrs,
    w_ptrs,
    offs_m,
    offs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if EVEN_M:
        a = tl.load(a_ptrs)
    else:
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
    if EVEN_N:
        w = tl.load(w_ptrs)
    else:
        w = tl.load(w_ptrs, mask=offs_n[None, :] < N, other=0.0)
    return a, w


@triton.jit
def _store_c(
    c_ptrs,
    c,
    offs_m,
    offs_n,
    M: tl.constexpr,
    N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    if EVEN_M:
        if EVEN_N:
            tl.store(c_ptrs, c)
        else:
            tl.store(c_ptrs, c, mask=offs_n[None, :] < N)
    else:
        if EVEN_N:
            tl.store(c_ptrs, c, mask=offs_m[:, None] < M)
        else:
            tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _tbmm_tail_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sw_b: tl.constexpr,
    sw_k: tl.constexpr,
    sw_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    n_mb: tl.constexpr,
    n_nb: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    TAIL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    """Two interleaved batches per program, BLOCK_K body plus an exact TAIL dot.

    Same structure as the general kernel's paired path, but the K reduction is
    split into full BLOCK_K blocks and one exact TAIL-row block instead of N
    equal blocks. For K = 192 that is one 128-row block and one 64-row block, so
    each weight request carries twice the bytes of the 64-row form while the
    bytes moved and the number of real multiply-adds are unchanged (no padding,
    no masking off real K rows).
    """
    pid = tl.program_id(0)
    per_b = n_mb * n_nb
    pid_b0 = (pid // per_b) * 2
    rem = pid % per_b
    pid_m = rem // n_nb
    pid_n = rem % n_nb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_t = tl.arange(0, max(TAIL, 1))

    a_row = offs_m[:, None] * sa_m
    k_col = offs_k[None, :] * sa_k
    w_row = offs_k[:, None] * sw_k
    n_col = offs_n[None, :] * sw_n
    c_row = offs_m[:, None] * sc_m
    c_col = offs_n[None, :] * sc_n

    abase0 = a_ptr + (pid_b0 * sa_b) + a_row
    wbase0 = w_ptr + (pid_b0 * sw_b) + n_col
    abase1 = a_ptr + ((pid_b0 + 1) * sa_b) + a_row
    wbase1 = w_ptr + ((pid_b0 + 1) * sw_b) + n_col

    a0_ptrs = abase0 + k_col
    w0_ptrs = wbase0 + w_row
    a1_ptrs = abase1 + k_col
    w1_ptrs = wbase1 + w_row
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K - TAIL, BLOCK_K):
        a0, w0 = _load_ab(a0_ptrs, w0_ptrs, offs_m, offs_n, M, N, EVEN_M, EVEN_N)
        acc0 = tl.dot(a0, w0, acc0)
        a0_ptrs += BLOCK_K * sa_k
        w0_ptrs += BLOCK_K * sw_k
        a1, w1 = _load_ab(a1_ptrs, w1_ptrs, offs_m, offs_n, M, N, EVEN_M, EVEN_N)
        acc1 = tl.dot(a1, w1, acc1)
        a1_ptrs += BLOCK_K * sa_k
        w1_ptrs += BLOCK_K * sw_k

    if TAIL > 0:
        t0 = (K - TAIL) * sa_k
        u0 = (K - TAIL) * sw_k
        a0, w0 = _load_ab(
            abase0 + t0 + offs_t[None, :] * sa_k,
            wbase0 + u0 + offs_t[:, None] * sw_k,
            offs_m,
            offs_n,
            M,
            N,
            EVEN_M,
            EVEN_N,
        )
        acc0 = tl.dot(a0, w0, acc0)
        a1, w1 = _load_ab(
            abase1 + t0 + offs_t[None, :] * sa_k,
            wbase1 + u0 + offs_t[:, None] * sw_k,
            offs_m,
            offs_n,
            M,
            N,
            EVEN_M,
            EVEN_N,
        )
        acc1 = tl.dot(a1, w1, acc1)

    c0_ptrs = c_ptr + (pid_b0 * sc_b) + c_row + c_col
    _store_c(
        c0_ptrs, acc0.to(c_ptr.dtype.element_ty), offs_m, offs_n, M, N, EVEN_M, EVEN_N
    )
    c1_ptrs = c_ptr + ((pid_b0 + 1) * sc_b) + c_row + c_col
    _store_c(
        c1_ptrs, acc1.to(c_ptr.dtype.element_ty), offs_m, offs_n, M, N, EVEN_M, EVEN_N
    )


@triton.jit
def _tbmm_bp_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sw_b: tl.constexpr,
    sw_k: tl.constexpr,
    sw_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    n_mb: tl.constexpr,
    n_nb: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Large-M path addressed with block descriptors instead of index tensors.

    The raw-pointer form rebuilds a [BLOCK_M, BLOCK_K] and a [BLOCK_K, BLOCK_N]
    element index tensor on every K step, and the profile shows a ~3.2 us Scalar
    term on both families that scales with the number of addressed elements.
    Block pointers describe the same tile by shape/stride scalars and are
    advanced by a scalar offset, so if that Scalar term is address ALU work it
    disappears; every dimension divides the tile exactly here and the innermost
    stride is 1, so no boundary checks are needed.
    """
    pid = tl.program_id(0)
    per_b = n_mb * n_nb
    pid_b = pid // per_b
    rem = pid % per_b
    pid_m = rem // n_nb
    pid_n = rem % n_nb

    a_bp = tl.make_block_ptr(
        base=a_ptr + pid_b * sa_b,
        shape=(M, K),
        strides=(sa_m, sa_k),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_bp = tl.make_block_ptr(
        base=w_ptr + pid_b * sw_b,
        shape=(K, N),
        strides=(sw_k, sw_n),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    c_bp = tl.make_block_ptr(
        base=c_ptr + pid_b * sc_b,
        shape=(M, N),
        strides=(sc_m, sc_n),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_bp), tl.load(w_bp), acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K))
        w_bp = tl.advance(w_bp, (BLOCK_K, 0))

    tl.store(c_bp, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _tbmm_bp2_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sw_b: tl.constexpr,
    sw_k: tl.constexpr,
    sw_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    n_mb: tl.constexpr,
    n_nb: tl.constexpr,
    units,
    B_TOT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Same tiles as _tbmm_bp_kernel, but each program walks `units` tile units.

    The profile of the one-unit-per-program form shows MTE1 (L1->L0 fill)
    3.9042 us + MAC 6.0987 us = 92.5% of a 10.8149 us aicore window, i.e. the
    window is the sum of the two pipes, and only 0.81 us of it is anything
    else. The launch is 64 programs on 32 cores, so every core runs two
    programs and pays two block prologues, two pipeline fills and one
    inter-program transition. Walking the two units inside one program removes
    those without touching fill volume, MAC work, tiles or the accumulator: the
    units are strictly sequential and `acc` is re-zeroed for each, so only one
    128 KiB fp32 accumulator is ever live (the concurrent two-accumulator form
    measured earlier duplicated the A fill and lost 0.8%).

    `units` is a runtime argument rather than a constexpr so the loop is a real
    `scf.for` and the two copies of the K loop cannot both be live at once.

    Units are enumerated batch-innermost. Every measured G1 output is a plain
    contiguous [B*M, N] matrix with row r = m*B + b, so pairing a program's two
    units by adjacent batch at a fixed m-tile (consecutive r) turns the two
    output rows into one contiguous 2*N-element run instead of two N-element
    runs M*N elements apart, and it also halves the number of distinct weight
    slabs the 32 concurrent programs are streaming. Fill volume, MAC count,
    grid and the accumulator are unchanged by the enumeration order.
    """
    pid = tl.program_id(0)
    for i in range(units):
        u = pid * units + i
        pid_b = u % B_TOT
        rest = u // B_TOT
        pid_m = rest // n_nb
        pid_n = rest % n_nb

        a_bp = tl.make_block_ptr(
            base=a_ptr + pid_b * sa_b,
            shape=(M, K),
            strides=(sa_m, sa_k),
            offsets=(pid_m * BLOCK_M, 0),
            block_shape=(BLOCK_M, BLOCK_K),
            order=(1, 0),
        )
        w_bp = tl.make_block_ptr(
            base=w_ptr + pid_b * sw_b,
            shape=(K, N),
            strides=(sw_k, sw_n),
            offsets=(0, pid_n * BLOCK_N),
            block_shape=(BLOCK_K, BLOCK_N),
            order=(1, 0),
        )
        c_bp = tl.make_block_ptr(
            base=c_ptr + pid_b * sc_b,
            shape=(M, N),
            strides=(sc_m, sc_n),
            offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in range(0, K, BLOCK_K):
            acc = tl.dot(tl.load(a_bp), tl.load(w_bp), acc)
            a_bp = tl.advance(a_bp, (0, BLOCK_K))
            w_bp = tl.advance(w_bp, (BLOCK_K, 0))

        tl.store(c_bp, acc.to(c_ptr.dtype.element_ty))


@triton.jit
def _tbmm_kernel(
    a_ptr,
    w_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    sa_b: tl.constexpr,
    sa_m: tl.constexpr,
    sa_k: tl.constexpr,
    sw_b: tl.constexpr,
    sw_k: tl.constexpr,
    sw_n: tl.constexpr,
    sc_b: tl.constexpr,
    sc_m: tl.constexpr,
    sc_n: tl.constexpr,
    n_mb: tl.constexpr,
    n_nb: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BPB: tl.constexpr,
):
    pid = tl.program_id(0)
    per_b = n_mb * n_nb
    pid_b0 = (pid // per_b) * BPB
    rem = pid % per_b
    pid_m = rem // n_nb
    pid_n = rem % n_nb

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_row = offs_m[:, None] * sa_m
    k_col = offs_k[None, :] * sa_k
    w_row = offs_k[:, None] * sw_k
    n_col = offs_n[None, :] * sw_n
    c_row = offs_m[:, None] * sc_m
    c_col = offs_n[None, :] * sc_n

    a0_ptrs = a_ptr + (pid_b0 * sa_b) + a_row + k_col
    w0_ptrs = w_ptr + (pid_b0 * sw_b) + w_row + n_col
    acc0 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if BPB == 2:
        # Second batch of this program: own operands and own accumulator, but a
        # shared K loop, so one prologue/fill/epilogue serves both batches.
        a1_ptrs = a_ptr + ((pid_b0 + 1) * sa_b) + a_row + k_col
        w1_ptrs = w_ptr + ((pid_b0 + 1) * sw_b) + w_row + n_col
        acc1 = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, K, BLOCK_K):
        a0, w0 = _load_ab(a0_ptrs, w0_ptrs, offs_m, offs_n, M, N, EVEN_M, EVEN_N)
        acc0 = tl.dot(a0, w0, acc0)
        a0_ptrs += BLOCK_K * sa_k
        w0_ptrs += BLOCK_K * sw_k
        if BPB == 2:
            a1, w1 = _load_ab(a1_ptrs, w1_ptrs, offs_m, offs_n, M, N, EVEN_M, EVEN_N)
            acc1 = tl.dot(a1, w1, acc1)
            a1_ptrs += BLOCK_K * sa_k
            w1_ptrs += BLOCK_K * sw_k

    c0_ptrs = c_ptr + (pid_b0 * sc_b) + c_row + c_col
    _store_c(
        c0_ptrs, acc0.to(c_ptr.dtype.element_ty), offs_m, offs_n, M, N, EVEN_M, EVEN_N
    )
    if BPB == 2:
        c1_ptrs = c_ptr + ((pid_b0 + 1) * sc_b) + c_row + c_col
        _store_c(
            c1_ptrs,
            acc1.to(c_ptr.dtype.element_ty),
            offs_m,
            offs_n,
            M,
            N,
            EVEN_M,
            EVEN_N,
        )


def _as_perm(p, ndim):
    """Normalize a perm argument to a tuple of ints (or None for identity)."""
    if p is None:
        return None
    if isinstance(p, (list, tuple)):
        return tuple(int(v) for v in p)
    if hasattr(p, "tolist"):
        return tuple(int(v) for v in p.tolist())
    return tuple(int(v) for v in p)


def _inv_perm(p):
    inv = [0] * len(p)
    for i, v in enumerate(p):
        inv[v] = i
    return tuple(inv)


def _pick_tiles(M, N, K):
    # Prefer the largest BLOCK_K that divides K: fewer unrolled loop bodies and
    # a longer contiguous run per A row, with no K masking.
    if (K % 128) == 0:
        block_k = 128
    elif (K % 64) == 0:
        block_k = 64
    elif (K % 32) == 0:
        block_k = 32
    else:
        block_k = 16
    if M >= 128:
        # One program owns a full N row block: A is read once instead of once
        # per n-block, and each (m, batch) output run is 2*N contiguous bytes.
        if N >= 256:
            return 128, 256, block_k
        return 128, 128, block_k
    # Small-M path: keep one batch's whole K x N weight slab assignment per
    # program, i.e. N is not split, so the per-program scalar/prologue cost is
    # amortised over the full row block and each weight row is a single
    # contiguous 2*N byte run.
    if N >= 512:
        block_n = 512
    else:
        if N >= 256:
            block_n = 256
        else:
            block_n = 128
    return 16, block_n, block_k


def _launch(
    input,
    weight,
    *,
    bias=None,
    scale=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=1,
):
    x1 = input
    x2 = weight
    if x1.dim() == 2:
        x1 = x1.unsqueeze(0)
    if x2.dim() == 2:
        x2 = x2.unsqueeze(0)

    p1 = _as_perm(perm_x1, x1.dim())
    p2 = _as_perm(perm_x2, x2.dim())
    p3 = _as_perm(perm_y, 3)
    if p1 is None:
        p1 = tuple(range(x1.dim()))
    if p2 is None:
        p2 = tuple(range(x2.dim()))
    if p3 is None:
        p3 = (0, 1, 2)

    a = x1.permute(p1)  # [B, M, K] view
    w = x2.permute(p2)  # [B, K, N] view
    batch, M, K = a.shape
    _, _, N = w.shape

    y_shape = (batch, M, N)
    out_shape = tuple(y_shape[p3[i]] for i in range(3))
    out = torch.empty(out_shape, device=x1.device, dtype=x1.dtype)
    # y is a [B, M, N] view of the freshly allocated output buffer, so the
    # kernel can write it through ordinary strides.
    y = out.permute(_inv_perm(p3))

    block_m, block_n, block_k = _pick_tiles(M, N, K)
    n_mb = triton.cdiv(M, block_m)
    n_nb = triton.cdiv(N, block_n)

    # Batches per program. On the tile-register-light small-M path one extra
    # batch shares the same prologue, pipeline fill and epilogue; on the large-M
    # path the accumulator is already at its practical size, so keep one batch
    # per program there.
    bpb = 2 if (block_m < 128 and (batch % 2) == 0) else 1
    grid = (triton.cdiv(batch, bpb) * n_mb * n_nb,)

    sa_b, sa_m, sa_k = a.stride()
    sw_b, sw_k, sw_n = w.stride()
    sc_b, sc_m, sc_n = y.stride()

    # Declared pipeline depth for the large-M descriptor kernel. The compile
    # log shows the backend derives its own multi-buffer mode
    # (--multibuffer-mode with l1=2), so the declared depth only changes how
    # much L1 the launch reserves: at 3 stages that is 3 * (32768 + 65536) =
    # 294912 B, at 2 stages 196608 B. Reserving less leaves the rest of the
    # 524288 B buffer budget for the in-flight weight slab and the store, so
    # the shallower declaration is tested here on the units=2 form.
    num_stages = 2

    # Paired small-M path with a wide K body: BLOCK_K=128 plus an exact tail dot
    # gives each weight request twice the bytes while moving the same data. Only
    # taken when the tail is a legal tile width, otherwise the general kernel
    # handles the shape.
    tail = K % 128 if K >= 128 else -1
    if block_m < 128 and (batch % 2) == 0 and tail in (0, 16, 32, 64):
        _tbmm_tail_kernel[grid](
            a,
            w,
            y,
            M,
            N,
            K,
            sa_b,
            sa_m,
            sa_k,
            sw_b,
            sw_k,
            sw_n,
            sc_b,
            sc_m,
            sc_n,
            n_mb,
            n_nb,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=128,
            TAIL=tail,
            EVEN_M=(M % block_m) == 0,
            EVEN_N=(N % block_n) == 0,
            num_stages=1,
        )
        return out

    # Large-M path: block-descriptor addressing when every dimension divides the
    # tile exactly and the innermost strides are 1 (no boundary handling needed).
    if (
        block_m >= 128
        and (M % block_m) == 0
        and (N % block_n) == 0
        and (K % block_k) == 0
        and sa_k == 1
        and sw_n == 1
        and sc_n == 1
    ):
        total_units = batch * n_mb * n_nb
        # Merge the two programs each core would otherwise run into one program
        # that walks both units sequentially, so a core pays one block prologue
        # and one inter-program transition instead of two. Only when the merge
        # keeps every core busy (a power-of-two split that still fills the 32
        # cores at least once).
        units = 2 if (total_units % 2) == 0 and (total_units // 2) >= 32 else 1
        if units == 2:
            _tbmm_bp2_kernel[(total_units // 2,)](
                a,
                w,
                y,
                M,
                N,
                K,
                sa_b,
                sa_m,
                sa_k,
                sw_b,
                sw_k,
                sw_n,
                sc_b,
                sc_m,
                sc_n,
                n_mb,
                n_nb,
                units,
                B_TOT=batch,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_K=block_k,
                num_stages=num_stages,
            )
            return out
        _tbmm_bp_kernel[(total_units,)](
            a,
            w,
            y,
            M,
            N,
            K,
            sa_b,
            sa_m,
            sa_k,
            sw_b,
            sw_k,
            sw_n,
            sc_b,
            sc_m,
            sc_n,
            n_mb,
            n_nb,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_stages=num_stages,
        )
        return out

    _tbmm_kernel[grid](
        a,
        w,
        y,
        M,
        N,
        K,
        sa_b,
        sa_m,
        sa_k,
        sw_b,
        sw_k,
        sw_n,
        sc_b,
        sc_m,
        sc_n,
        n_mb,
        n_nb,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_M=(M % block_m) == 0,
        EVEN_N=(N % block_n) == 0,
        BPB=bpb,
        num_stages=num_stages,
    )
    return out


def npu_transpose_batchmatmul(
    input,
    weight,
    *,
    bias=None,
    scale=None,
    perm_x1=None,
    perm_x2=None,
    perm_y=None,
    batch_split_factor=1,
):
    """BF16 batched matrix multiplication with input and output token/batch transposes."""
    if (
        input.ndim != 3
        or weight.ndim != 3
        or not input.is_contiguous()
        or not weight.is_contiguous()
        or input.dtype != torch.bfloat16
        or weight.dtype != input.dtype
    ):
        raise ValueError("expected contiguous BF16 rank-three tensors")
    if (
        bias is not None
        or scale is not None
        or batch_split_factor != 1
        or tuple(perm_x1 or ()) != (1, 0, 2)
        or tuple(perm_x2 or (0, 1, 2)) != (0, 1, 2)
        or tuple(perm_y or ()) != (1, 0, 2)
    ):
        raise ValueError("expected MBK x BKN -> MBN without bias or scale")
    if input.shape[1:] != weight.shape[:2] or not input.numel() or not weight.numel():
        raise ValueError("incompatible or empty batched matrices")
    if input.device.type != "npu" or weight.device != input.device:
        raise ValueError("expected inputs on the same NPU")
    with torch_device_fn.device(input.device):
        return _launch(
            input,
            weight,
            bias=bias,
            scale=scale,
            perm_x1=perm_x1,
            perm_x2=perm_x2,
            perm_y=perm_y,
            batch_split_factor=batch_split_factor,
        )
