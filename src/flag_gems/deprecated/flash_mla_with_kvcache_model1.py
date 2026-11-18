import logging

import torch
import triton
import triton.experimental.tle.language as tle
import triton.language as tl
import triton.language.core as tlc
from triton.experimental.tle.language.gpu import types as tle_types
from triton.runtime.jit import constexpr_function
from triton.tools.tensor_descriptor import TensorDescriptor

from flag_gems.utils.device_info import get_device_capability

logger = logging.getLogger(__name__)

aggregate = tlc._aggregate

# Runtime scalar aggregate fields are usually tl.tensor, but triton
# specializes some int kernel args (e.g. value 1) into plain python ints, so
# scalar fields must accept constexpr too (wrapped by _as_value).
VALUE = (tlc.tensor, tlc.constexpr)


def _as_value(x):
    return tl.constexpr(x) if isinstance(x, (int, float, bool)) else x


# Only called at compile time from @constexpr_function constructors; marking
# it builtin makes the jit dependency scanner skip it.
_as_value.__triton_builtin__ = True

# MODEL1 geometry. These are tl.constexpr so device code can reference them
# directly (FlagTree jit rejects plain int globals); host code uses .value.
FULL_HEAD_DIM = tl.constexpr(512)  # d_qk == d_v
HALF_HEAD_DIM = tl.constexpr(256)  # consumer0 owns [0:256), consumer1 [256:512)
NOPE_DIM = tl.constexpr(448)  # fp8 NoPE dims; [448:512) is the bf16 RoPE part
ROPE_DIM = tl.constexpr(64)  # FULL_HEAD_DIM - NOPE_DIM
DEQUANT_GROUP = tl.constexpr(64)  # columns the producer dequantizes per step
BLOCK_HEADS = tl.constexpr(64)  # Q/O rows per head-block tile
BLOCK_TOKENS = tl.constexpr(64)  # KV tokens per block (FlashMLA TOPK_BLOCK_SIZE)
TOKEN_BYTES = tl.constexpr(576)  # fp8 NoPE (448 B) + bf16 RoPE (128 B) per token
PAGE_TOKEN_BYTES = 584  # per-token page footprint incl. the 8 scale bytes
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)

# setmaxnreg budgets: consumer0 needs the largest budget (it holds the f32
# accumulator plus the softmax tiles), the producer takes what is left.
# Blocks per batch-head below which the balanced device-side schedule costs
# more (metadata + combine kernels) than the imbalance it removes.
MIN_BLOCKS_FOR_DYNAMIC_SCHED = 8

DEFAULT_C0_REGS = 240
DEFAULT_C1_REGS = 168


def _alloc_fn(size: int, align: int, stream):
    _ = align
    _ = stream
    return torch.empty(size, dtype=torch.int8, device="cuda")


@tlc.builtin
def tle_subslice(buf, offsets, shape, _semantic=None):
    """Column/row subslice VIEW of a shared buffered_tensor. The view keeps the
    parent layout + alloc_shape so a wgmma reading it sees the same swizzle as
    the full buffer. Safe as a wgmma operand and as a tl.store destination;
    NOT safe as a TMA-copy destination."""
    offsets = [int(tlc._unwrap_if_constexpr(o)) for o in offsets]
    shape = [int(tlc._unwrap_if_constexpr(s)) for s in shape]
    result_ty = tle_types.buffered_tensor_type(
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )
    handle = _semantic.builder.create_memdesc_subslice(
        result_ty.to_ir(_semantic.builder), buf.handle, offsets
    )
    return tle_types.buffered_tensor(
        handle,
        buf.dtype,
        shape,
        buf.type.storage,
        buf.type.layout,
        _semantic,
        alloc_shape=buf.type.alloc_shape,
    )


# ============================================================================
# Aggregates (used by the producer / default partition only -- the consumers
# stay flat, see the module docstring)
# ============================================================================


@aggregate
class Config:
    """Schedule/geometry scalars for the producer's segment walk."""

    NUM_BLOCKS: VALUE  # static (padded) KV blocks per batch-head item
    NUM_MAIN_BLOCKS: VALUE  # of those, the ones from the main pool
    BLOCKS_PER_CTA: VALUE  # static schedule: blocks per CTA
    NUM_LONG_CTAS: VALUE  # CTAs that get one extra block (the remainder)
    NUM_BH: VALUE  # batch-head work items = B * SEQ_Q * NUM_HEAD_BLOCKS
    NUM_Q_HEADS: tl.constexpr
    SEQ_Q: tl.constexpr
    NUM_HEAD_BLOCKS: tl.constexpr
    NUM_KV_BUFS: tl.constexpr
    DYNAMIC_SCHED: tl.constexpr  # dynamic lengths -> schedule in sched_ptr

    @constexpr_function
    def __init__(
        self,
        NUM_BLOCKS,
        NUM_MAIN_BLOCKS,
        BLOCKS_PER_CTA,
        NUM_LONG_CTAS,
        NUM_BH,
        NUM_Q_HEADS,
        SEQ_Q,
        NUM_KV_BUFS,
        DYNAMIC_SCHED,
    ):
        self.NUM_BLOCKS = _as_value(NUM_BLOCKS)
        self.NUM_MAIN_BLOCKS = _as_value(NUM_MAIN_BLOCKS)
        self.BLOCKS_PER_CTA = _as_value(BLOCKS_PER_CTA)
        self.NUM_LONG_CTAS = _as_value(NUM_LONG_CTAS)
        self.NUM_BH = _as_value(NUM_BH)
        self.NUM_Q_HEADS = tl.constexpr(NUM_Q_HEADS)
        self.SEQ_Q = tl.constexpr(SEQ_Q)
        self.NUM_HEAD_BLOCKS = tl.constexpr(NUM_Q_HEADS // BLOCK_HEADS)
        self.NUM_KV_BUFS = tl.constexpr(NUM_KV_BUFS)
        self.DYNAMIC_SCHED = tl.constexpr(DYNAMIC_SCHED)

    @triton.jit
    def bh_coords(self, bh_idx):
        """bh_idx -> (batch_idx, seq_q_idx, head_block_idx)."""
        batch_idx = bh_idx // (self.SEQ_Q * self.NUM_HEAD_BLOCKS)
        rest = bh_idx % (self.SEQ_Q * self.NUM_HEAD_BLOCKS)
        return batch_idx, rest // self.NUM_HEAD_BLOCKS, rest % self.NUM_HEAD_BLOCKS


@aggregate
class Cache:
    """One paged MODEL1 KV pool: flat views, indices and optional per-batch
    dynamic length. Length semantics must stay in sync with
    _seg_valid_blocks (the consumers' flat twin)."""

    fp8_base: tl.tensor  # flat fp8 view of the whole pool
    scale_i64_base: tl.tensor  # flat int64 view (8 packed E8M0 scale bytes)
    bf16_base: tl.tensor  # flat bf16 view (RoPE part)
    indices_ptr: tl.tensor  # [B, SEQ_Q, topk] int32 token indices
    length_ptr: tl.tensor  # [B] int32 (dummy when HAVE_LENGTH is False)
    static_length: VALUE  # topk / extra_topk: the length when static
    num_blocks: VALUE  # static block count of this pool (0 = no pool)
    stride_page: VALUE  # bytes per cache page
    page_size: VALUE  # tokens per cache page
    num_tokens: VALUE  # total tokens in the pool (index bound)
    stride_indices_batch: VALUE
    stride_indices_seq: VALUE
    MIN_VALID_BLOCKS: tl.constexpr  # 1 for the main pool (so every batch-head
    HAVE_LENGTH: tl.constexpr  # writes out/lse), 0 for the extra pool

    @constexpr_function
    def __init__(
        self,
        fp8_base,
        scale_i64_base,
        bf16_base,
        indices_ptr,
        length_ptr,
        static_length,
        num_blocks,
        stride_page,
        page_size,
        num_tokens,
        stride_indices_batch,
        stride_indices_seq,
        MIN_VALID_BLOCKS,
        HAVE_LENGTH,
    ):
        self.fp8_base = fp8_base
        self.scale_i64_base = scale_i64_base
        self.bf16_base = bf16_base
        self.indices_ptr = indices_ptr
        self.length_ptr = length_ptr
        self.static_length = _as_value(static_length)
        self.num_blocks = _as_value(num_blocks)
        self.stride_page = _as_value(stride_page)
        self.page_size = _as_value(page_size)
        self.num_tokens = _as_value(num_tokens)
        self.stride_indices_batch = _as_value(stride_indices_batch)
        self.stride_indices_seq = _as_value(stride_indices_seq)
        self.MIN_VALID_BLOCKS = tl.constexpr(MIN_VALID_BLOCKS)
        self.HAVE_LENGTH = tl.constexpr(HAVE_LENGTH)

    @triton.jit
    def kv_len(self, batch_idx):
        if self.HAVE_LENGTH:
            return tl.load(self.length_ptr + batch_idx)
        else:
            return self.static_length

    @triton.jit
    def valid_blocks(self, batch_idx):
        """Blocks of this pool that actually carry data for batch_idx."""
        if self.HAVE_LENGTH:
            length = tl.load(self.length_ptr + batch_idx)
            n = (length + BLOCK_TOKENS - 1) // BLOCK_TOKENS
            return tl.minimum(tl.maximum(n, self.MIN_VALID_BLOCKS), self.num_blocks)
        else:
            return self.num_blocks

    @triton.jit
    def token_base(self, batch_idx, seq_q_idx):
        return (
            self.indices_ptr
            + batch_idx * self.stride_indices_batch
            + seq_q_idx * self.stride_indices_seq
        )


@aggregate
class Channel:
    """All smem buffers + barriers of the producer/consumer pipeline. The
    buffers themselves are allocated in the kernel body (an aggregate method
    would stay a non-inlined tt.call)."""

    q: tle_types.buffered_tensor  # [1, 64, 512] bf16: Q, reused to stage O
    kv: tle_types.buffered_tensor  # [NUM_KV_BUFS, 64, 512] bf16: dequantized KV
    mask: tle_types.buffered_tensor  # [NUM_KV_BUFS, 64] f32: QK additive mask
    # P = the softmax probabilities tile of the current block; consumer0
    # computes it and hands it (with the online-softmax rescale factor alpha)
    # to consumer1 through NUM_P_BUFS double-buffered smem slots.
    p: tle_types.buffered_tensor  # [NUM_P_BUFS, 64, 64] bf16
    alpha: tle_types.buffered_tensor  # [NUM_P_BUFS, 64] f32
    o_scale: tle_types.buffered_tensor  # [1, 64] f32: O rescale factor
    q_empty: tle_types.barrier
    q_full: tle_types.barrier
    kv_full: tle_types.barrier
    kv_empty: tle_types.barrier
    p_full: tle_types.barrier
    p_empty: tle_types.barrier
    o_scale_full: tle_types.barrier
    o_scale_empty: tle_types.barrier
    o_ready: tle_types.barrier

    @constexpr_function
    def __init__(
        self,
        q,
        kv,
        mask,
        p,
        alpha,
        o_scale,
        q_empty,
        q_full,
        kv_full,
        kv_empty,
        p_full,
        p_empty,
        o_scale_full,
        o_scale_empty,
        o_ready,
    ):
        self.q = q
        self.kv = kv
        self.mask = mask
        self.p = p
        self.alpha = alpha
        self.o_scale = o_scale
        self.q_empty = q_empty
        self.q_full = q_full
        self.kv_full = kv_full
        self.kv_empty = kv_empty
        self.p_full = p_full
        self.p_empty = p_empty
        self.o_scale_full = o_scale_full
        self.o_scale_empty = o_scale_empty
        self.o_ready = o_ready


# ============================================================================
# Schedule walk (shared by all partitions)
# ============================================================================


@triton.jit
def _cta_of_block(block_idx, blocks_per_cta, num_long_ctas):
    # inverse of the range-start formula: which CTA owns valid block block_idx
    threshold = num_long_ctas * (blocks_per_cta + 1)
    return tl.where(
        block_idx < threshold,
        block_idx // (blocks_per_cta + 1),
        num_long_ctas + (block_idx - threshold) // blocks_per_cta,
    )


@triton.jit
def _seg_valid_blocks(
    batch_idx,
    topk_len_ptr,
    extra_topk_len_ptr,
    TOPK,
    EXTRA_TOPK,
    NUM_MAIN_BLOCKS,
    NUM_EXTRA_BLOCKS,
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
):
    """Per-batch-head VALID block counts (main, extra). Blocks entirely beyond
    the dynamic lengths are excluded from the schedule; the main part keeps at
    least one block so every batch-head writes its out/lse. Flat twin of
    Cache.valid_blocks, used by the (flat) consumer partitions."""
    if HAVE_TOPK_LEN:
        main_len = tl.load(topk_len_ptr + batch_idx)
        n_main = tl.minimum(
            tl.maximum((main_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS, 1),
            NUM_MAIN_BLOCKS,
        )
    else:
        n_main = NUM_MAIN_BLOCKS
    if HAVE_EXTRA:
        if HAVE_EXTRA_TOPK_LEN:
            extra_len = tl.load(extra_topk_len_ptr + batch_idx)
            n_extra = tl.minimum(
                tl.maximum((extra_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS, 0),
                NUM_EXTRA_BLOCKS,
            )
        else:
            n_extra = NUM_EXTRA_BLOCKS
    else:
        n_extra = 0
    return n_main, n_extra


@triton.jit
def _walk_init(
    sched_ptr,
    NUM_BLOCKS,
    BLOCKS_PER_CTA,
    NUM_LONG_CTAS,
    NUM_BH,
    DYNAMIC_SCHED: tl.constexpr,
):
    """Per-CTA start state of the segment walk over the VALID block space.
    Dynamic: read the device-computed schedule, laid out as
    [0..NUM_BH) exclusive prefix, [NUM_BH] total, [NUM_BH+1] blocks_per_cta,
    [NUM_BH+2] num_long_ctas, then start bh_idx[P] and start blk_in_bh[P].
    Static: the valid space equals the full space, derive arithmetically."""
    pid = tl.program_id(0)
    if DYNAMIC_SCHED:
        blocks_per_cta = tl.load(sched_ptr + NUM_BH + 1)
        num_long_ctas = tl.load(sched_ptr + NUM_BH + 2)
    else:
        blocks_per_cta = BLOCKS_PER_CTA
        num_long_ctas = NUM_LONG_CTAS
    blk = pid * blocks_per_cta + min(pid, num_long_ctas)
    blk_end = blk + blocks_per_cta + tl.where(pid < num_long_ctas, 1, 0)
    if DYNAMIC_SCHED:
        num_ctas = tl.num_programs(0)
        bh_idx = tl.load(sched_ptr + NUM_BH + 3 + pid)
        blk_in_bh = tl.load(sched_ptr + NUM_BH + 3 + num_ctas + pid)
    else:
        bh_idx = blk // NUM_BLOCKS
        blk_in_bh = blk - bh_idx * NUM_BLOCKS
    return blk, blk_end, bh_idx, blk_in_bh


# ============================================================================
# Producer (default partition -- takes the aggregate bundles)
# ============================================================================


@triton.jit
def _scale_from_u64_bf16(scales_u64, i: tl.constexpr):
    # E8M0 -> 2^(e-127) built directly in bf16 (its exponent field is bits
    # [7:15), so e << 7). bf16 shares the f32 exponent/bias, so 2^k is exact
    # and the per-element scale multiply stays bf16x2-packed instead of f32.
    e = (scales_u64 >> (8 * i)).to(tl.int32) & 0xFF
    return (e << 7).to(tl.uint16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _dequant_store(
    kv_slot,
    mask_slot,
    token_base,
    blk_in_pool,
    kv_len,
    fp8_base,
    scale_i64_base,
    bf16_base,
    stride_page,
    page_size,
    num_tokens,
):
    """Gather + dequantize one 64-token block into the single [64, 512] KV
    buffer as eight [64, 64] column-group subslices in a fully unrolled static
    loop: 7 fp8 NoPE groups (each with its own E8M0 scale) + 1 RoPE group.
    Streaming one group at a time keeps the peak live tile ~16 regs/thread
    instead of 64, which is what keeps the producer spill-free.

    Also publishes the per-token QK additive mask (0 valid, -inf otherwise).

    All value loads are masked: tokens beyond kv_len or with invalid ids may
    be NaN-poisoned in the cache, and NaN * 0 would poison the PV matmul.
    """
    token_offs = tl.arange(0, BLOCK_TOKENS)
    col_offs = tl.arange(0, DEQUANT_GROUP)
    pool_offs = blk_in_pool * BLOCK_TOKENS + token_offs
    ids = tl.load(token_base + pool_offs)
    valid = (pool_offs < kv_len) & (ids >= 0) & (ids < num_tokens)
    # invalid tokens' KV is zeroed below, so qk == 0 there; +(-inf) => -inf.
    tl.store(tle.gpu.local_ptr(mask_slot), tl.where(valid, 0.0, float("-inf")))
    ids = tl.where(valid, ids, 0)
    page = ids // page_size
    tok_in_page = ids - page * page_size
    token_byte = page.to(tl.int64) * stride_page + tok_in_page * TOKEN_BYTES
    # all 8 E8M0 scale bytes in one 64-bit load (the scale row is 8B-aligned)
    scale_off = (
        page.to(tl.int64) * stride_page + page_size * TOKEN_BYTES
    ) // 8 + tok_in_page
    scales_u64 = tl.load(scale_i64_base + scale_off)

    valid_col = valid[:, None]
    group_cols = col_offs[None, :]
    for group_idx in tl.static_range(NOPE_DIM // DEQUANT_GROUP):
        fp8_group = tl.load(
            fp8_base + token_byte[:, None] + (group_idx * DEQUANT_GROUP + group_cols),
            mask=valid_col,
            other=0.0,
        ).to(tl.bfloat16)
        scale_bcast = tl.broadcast_to(
            _scale_from_u64_bf16(scales_u64, group_idx)[:, None],
            [BLOCK_TOKENS, DEQUANT_GROUP],
        )
        # packed bf16x2 multiply (triton emits scalar mul.bf16 otherwise)
        dequant = tl.inline_asm_elementwise(
            "mul.rn.bf16x2 $0, $1, $2;",
            "=r,r,r",
            [fp8_group, scale_bcast],
            dtype=tl.bfloat16,
            is_pure=True,
            pack=2,
        )
        tl.store(
            tle.gpu.local_ptr(
                tle_subslice(
                    kv_slot,
                    [0, group_idx * DEQUANT_GROUP],
                    [BLOCK_TOKENS, DEQUANT_GROUP],
                )
            ),
            dequant,
        )
    # RoPE group: dims [448:512) bf16, element index (token_byte >> 1) + column
    rope = tl.load(
        bf16_base + (token_byte >> 1)[:, None] + (NOPE_DIM // 2 + group_cols),
        mask=valid_col,
        other=0.0,
    )
    tl.store(
        tle.gpu.local_ptr(
            tle_subslice(kv_slot, [0, NOPE_DIM], [BLOCK_TOKENS, ROPE_DIM])
        ),
        rope,
    )


@triton.jit
def _model1_producer(desc_q, sched_ptr, cfg, ch, main_cache, extra_cache):
    blk, blk_end, bh_idx, blk_in_bh = _walk_init(
        sched_ptr,
        cfg.NUM_BLOCKS,
        cfg.BLOCKS_PER_CTA,
        cfg.NUM_LONG_CTAS,
        cfg.NUM_BH,
        cfg.DYNAMIC_SCHED,
    )
    seg_idx = 0
    kv_cnt = 0
    while blk < blk_end:
        batch_idx, seq_q_idx, head_block_idx = cfg.bh_coords(bh_idx)
        # With the dynamic schedule every scheduled block carries data, so a
        # segment spans the VALID blocks; with the static schedule it spans the
        # padded blocks and out-of-range tokens are masked per block instead.
        if cfg.DYNAMIC_SCHED:
            n_main = main_cache.valid_blocks(batch_idx)
            n_seg = n_main + extra_cache.valid_blocks(batch_idx)
        else:
            n_main = cfg.NUM_MAIN_BLOCKS
            n_seg = cfg.NUM_BLOCKS
        seg_len = min(n_seg - blk_in_bh, blk_end - blk)

        # Q for this segment via a single wide TMA into one [64, 512] buffer
        q_row0 = (batch_idx * cfg.SEQ_Q + seq_q_idx) * cfg.NUM_Q_HEADS + (
            head_block_idx * BLOCK_HEADS
        )
        tle.gpu.barrier_wait(ch.q_empty[0], phaseIdx=seg_idx)
        tle.gpu.copy(
            desc_q,
            ch.q.slot(0),
            [BLOCK_HEADS, FULL_HEAD_DIM],
            [q_row0, 0],
            barrier=ch.q_full[0],
        )

        main_tokens = main_cache.token_base(batch_idx, seq_q_idx)
        extra_tokens = extra_cache.token_base(batch_idx, seq_q_idx)
        main_kv_len = main_cache.kv_len(batch_idx)
        extra_kv_len = extra_cache.kv_len(batch_idx)

        for i in range(seg_len):
            cur_blk = blk_in_bh + i
            buf = kv_cnt % cfg.NUM_KV_BUFS
            cycle = kv_cnt // cfg.NUM_KV_BUFS
            tle.gpu.barrier_wait(ch.kv_empty[buf], phaseIdx=cycle)
            # store-into-subslice is safe here; TMA-into-subslice is not
            if cur_blk < n_main:
                _dequant_store(
                    ch.kv.slot(buf),
                    ch.mask.slot(buf),
                    main_tokens,
                    cur_blk,
                    main_kv_len,
                    main_cache.fp8_base,
                    main_cache.scale_i64_base,
                    main_cache.bf16_base,
                    main_cache.stride_page,
                    main_cache.page_size,
                    main_cache.num_tokens,
                )
            else:
                _dequant_store(
                    ch.kv.slot(buf),
                    ch.mask.slot(buf),
                    extra_tokens,
                    cur_blk - n_main,
                    extra_kv_len,
                    extra_cache.fp8_base,
                    extra_cache.scale_i64_base,
                    extra_cache.bf16_base,
                    extra_cache.stride_page,
                    extra_cache.page_size,
                    extra_cache.num_tokens,
                )
            tle.gpu.barrier_arrive(ch.kv_full[buf], phaseIdx=cycle)
            kv_cnt += 1

        blk += seg_len
        bh_idx += 1
        blk_in_bh = 0
        seg_idx += 1


# ============================================================================
# Consumers (worker partitions -- flat args; their argument lists ARE the
# warp_specialize capture list, whose order feeds register allocation)
# ============================================================================


@triton.jit
def _model1_consumer0(
    desc_o,
    o_ptr,
    attn_sink_ptr,
    topk_len_ptr,
    extra_topk_len_ptr,
    sched_ptr,
    lse_ptr,
    o_part_ptr,
    lse_part_ptr,
    sm_scale,
    TOPK,
    EXTRA_TOPK,
    NUM_MAIN_BLOCKS,
    NUM_BLOCKS,
    BLOCKS_PER_CTA,
    NUM_LONG_CTAS,
    NUM_BH,
    stride_lse_batch,
    stride_lse_head,
    q,
    kv,
    mask_smem,
    p_smem,
    alpha_smem,
    o_scale_smem,
    q_empty,
    q_full,
    kv_full,
    kv_empty,
    p_full,
    p_empty,
    o_scale_full,
    o_scale_empty,
    o_ready,
    NUM_Q_HEADS: tl.constexpr,
    SEQ_Q: tl.constexpr,
    HAVE_SINK: tl.constexpr,
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
    NUM_KV_BUFS: tl.constexpr,
    NUM_P_BUFS: tl.constexpr,
    DYNAMIC_SCHED: tl.constexpr,
    PARTITION_ID: tl.constexpr,
):
    NUM_HEAD_BLOCKS: tl.constexpr = NUM_Q_HEADS // BLOCK_HEADS
    pid = tl.program_id(0)
    head_offs = tl.arange(0, BLOCK_HEADS)
    dim_offs = tl.arange(0, HALF_HEAD_DIM)

    blk, blk_end, bh_idx, blk_in_bh = _walk_init(
        sched_ptr, NUM_BLOCKS, BLOCKS_PER_CTA, NUM_LONG_CTAS, NUM_BH, DYNAMIC_SCHED
    )
    seg_idx = 0
    kv_cnt = 0
    p_cnt = 0
    while blk < blk_end:
        batch_idx = bh_idx // (SEQ_Q * NUM_HEAD_BLOCKS)
        rest = bh_idx % (SEQ_Q * NUM_HEAD_BLOCKS)
        seq_q_idx = rest // NUM_HEAD_BLOCKS
        head_block_idx = rest % NUM_HEAD_BLOCKS
        if DYNAMIC_SCHED:
            n_main, n_extra = _seg_valid_blocks(
                batch_idx,
                topk_len_ptr,
                extra_topk_len_ptr,
                TOPK,
                EXTRA_TOPK,
                NUM_MAIN_BLOCKS,
                NUM_BLOCKS - NUM_MAIN_BLOCKS,
                HAVE_TOPK_LEN,
                HAVE_EXTRA,
                HAVE_EXTRA_TOPK_LEN,
            )
            n_valid = n_main + n_extra
        else:
            n_valid = NUM_BLOCKS
        seg_len = min(n_valid - blk_in_bh, blk_end - blk)

        m_i = tl.full([BLOCK_HEADS], -1e30, tl.float32)
        l_i = tl.zeros([BLOCK_HEADS], tl.float32)
        acc = tl.zeros([BLOCK_HEADS, HALF_HEAD_DIM], tl.float32)

        tle.gpu.barrier_wait(q_full[0], phaseIdx=seg_idx)

        # every scheduled block is valid (the schedule already excludes blocks
        # beyond the dynamic lengths), so this loop is branch-free and takes
        # its QK mask from smem instead of re-loading indices from global
        for i in range(seg_len):
            buf = kv_cnt % NUM_KV_BUFS
            cycle = kv_cnt // NUM_KV_BUFS
            tle.gpu.barrier_wait(kv_full[buf], phaseIdx=cycle)
            # QK over the full 512 dims in ONE wgmma (single KV buffer)
            qk = tle.gpu.wgmma(
                q.slot(0), kv.slot(buf), out_dtype=tl.float32, trans_b=True
            )
            qk = tle.gpu.wgmma_wait(0, qk)

            addmask = tl.load(tle.gpu.local_ptr(mask_smem.slot(buf)))
            qk = qk * (sm_scale * LOG2E) + addmask[None, :]

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            m_i = m_new
            p_bf16 = p.to(tl.bfloat16)

            # PV for output dims [0:256): P stays in registers (RS-mode), V is
            # the lo column VIEW of the KV buffer. Issue it first, then publish
            # P/alpha so the smem stores overlap the PV wgmma latency.
            acc = acc * alpha[:, None]
            acc = tle.gpu.wgmma(
                p_bf16,
                tle_subslice(kv.slot(buf), [0, 0], [BLOCK_TOKENS, HALF_HEAD_DIM]),
                acc,
            )

            p_slot = p_cnt % NUM_P_BUFS
            p_cycle = p_cnt // NUM_P_BUFS
            tle.gpu.barrier_wait(p_empty[p_slot], phaseIdx=p_cycle)
            tl.store(tle.gpu.local_ptr(alpha_smem.slot(p_slot)), alpha)
            tl.store(tle.gpu.local_ptr(p_smem.slot(p_slot)), p_bf16)
            tle.gpu.barrier_arrive(p_full[p_slot], phaseIdx=p_cycle)
            p_cnt += 1

            acc = tle.gpu.wgmma_wait(0, acc)
            tle.gpu.barrier_arrive(kv_empty[buf], phaseIdx=cycle)
            kv_cnt += 1

        # epilogue: lse excludes the sink; the sink only rescales O
        empty = l_i == 0.0
        l_safe = tl.where(empty, 1.0, l_i)
        full_bh = seg_len == n_valid
        if full_bh:
            lse = tl.where(empty, float("inf"), m_i * LN2 + tl.log(l_safe))
            tl.store(
                lse_ptr
                + batch_idx * stride_lse_batch
                + (head_block_idx * BLOCK_HEADS + head_offs) * stride_lse_head
                + seq_q_idx,
                lse,
            )
            if HAVE_SINK:
                sink = tl.load(attn_sink_ptr + head_block_idx * BLOCK_HEADS + head_offs)
                exp_m = tl.math.exp2(m_i)
                factor = exp_m / (exp_m * l_i + tl.exp(sink))
            else:
                factor = 1.0 / l_safe
            factor = tl.where(empty, 0.0, factor)
        else:
            lse_part = tl.where(empty, float("-inf"), m_i * LN2 + tl.log(l_safe))
            # a CTA has at most two partial segments: its first (slot 0) and
            # its last (slot 1); middle segments always cover a whole bh
            part_row = (pid * 2 + tl.where(seg_idx == 0, 0, 1)) * BLOCK_HEADS
            tl.store(lse_part_ptr + part_row + head_offs, lse_part)
            factor = tl.where(empty, 0.0, 1.0 / l_safe)

        # publish the O rescale factor to consumer1
        tle.gpu.barrier_wait(o_scale_empty[0], phaseIdx=seg_idx)
        tl.store(tle.gpu.local_ptr(o_scale_smem.slot(0)), factor)
        tle.gpu.barrier_arrive(o_scale_full[0], phaseIdx=seg_idx)

        out = acc * factor[:, None]
        if full_bh:
            # stage O[0:256) into the free Q smem lo half, wait for consumer1's
            # hi half, then TMA-store the full [64, 512] O
            q_row0 = (batch_idx * SEQ_Q + seq_q_idx) * NUM_Q_HEADS + (
                head_block_idx * BLOCK_HEADS
            )
            tl.store(
                tle.gpu.local_ptr(
                    tle_subslice(q.slot(0), [0, 0], [BLOCK_HEADS, HALF_HEAD_DIM])
                ),
                out.to(tl.bfloat16),
            )
            tle.gpu.barrier_wait(o_ready[0], phaseIdx=seg_idx)
            tle.gpu.copy(q.slot(0), desc_o, [BLOCK_HEADS, FULL_HEAD_DIM], [q_row0, 0])
        else:
            tle.gpu.barrier_wait(o_ready[0], phaseIdx=seg_idx)
            part_row = (pid * 2 + tl.where(seg_idx == 0, 0, 1)) * BLOCK_HEADS
            o_base = (
                o_part_ptr
                + (part_row + head_offs[:, None]).to(tl.int64) * FULL_HEAD_DIM
            )
            tl.store(o_base + dim_offs[None, :], out)

        tle.gpu.barrier_arrive(q_empty[0], phaseIdx=seg_idx)
        blk += seg_len
        bh_idx += 1
        blk_in_bh = 0
        seg_idx += 1


@triton.jit
def _model1_consumer1(
    desc_o,
    o_ptr,
    topk_len_ptr,
    extra_topk_len_ptr,
    sched_ptr,
    o_part_ptr,
    TOPK,
    EXTRA_TOPK,
    NUM_MAIN_BLOCKS,
    NUM_BLOCKS,
    BLOCKS_PER_CTA,
    NUM_LONG_CTAS,
    NUM_BH,
    q,
    kv,
    p_smem,
    alpha_smem,
    o_scale_smem,
    kv_full,
    kv_empty,
    p_full,
    p_empty,
    o_scale_full,
    o_scale_empty,
    o_ready,
    NUM_Q_HEADS: tl.constexpr,
    SEQ_Q: tl.constexpr,
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
    NUM_KV_BUFS: tl.constexpr,
    NUM_P_BUFS: tl.constexpr,
    DYNAMIC_SCHED: tl.constexpr,
    PARTITION_ID: tl.constexpr,
):
    NUM_HEAD_BLOCKS: tl.constexpr = NUM_Q_HEADS // BLOCK_HEADS
    pid = tl.program_id(0)
    head_offs = tl.arange(0, BLOCK_HEADS)
    dim_offs = tl.arange(0, HALF_HEAD_DIM)

    blk, blk_end, bh_idx, blk_in_bh = _walk_init(
        sched_ptr, NUM_BLOCKS, BLOCKS_PER_CTA, NUM_LONG_CTAS, NUM_BH, DYNAMIC_SCHED
    )
    seg_idx = 0
    kv_cnt = 0
    p_cnt = 0
    while blk < blk_end:
        batch_idx = bh_idx // (SEQ_Q * NUM_HEAD_BLOCKS)
        if DYNAMIC_SCHED:
            n_main, n_extra = _seg_valid_blocks(
                batch_idx,
                topk_len_ptr,
                extra_topk_len_ptr,
                TOPK,
                EXTRA_TOPK,
                NUM_MAIN_BLOCKS,
                NUM_BLOCKS - NUM_MAIN_BLOCKS,
                HAVE_TOPK_LEN,
                HAVE_EXTRA,
                HAVE_EXTRA_TOPK_LEN,
            )
            n_valid = n_main + n_extra
        else:
            n_valid = NUM_BLOCKS
        seg_len = min(n_valid - blk_in_bh, blk_end - blk)

        acc = tl.zeros([BLOCK_HEADS, HALF_HEAD_DIM], tl.float32)

        for i in range(seg_len):
            buf = kv_cnt % NUM_KV_BUFS
            cycle = kv_cnt // NUM_KV_BUFS
            tle.gpu.barrier_wait(kv_full[buf], phaseIdx=cycle)
            p_slot = p_cnt % NUM_P_BUFS
            p_cycle = p_cnt // NUM_P_BUFS
            tle.gpu.barrier_wait(p_full[p_slot], phaseIdx=p_cycle)
            alpha = tl.load(tle.gpu.local_ptr(alpha_smem.slot(p_slot)))
            # PV for output dims [256:512): P from smem (SS-mode), V is the hi
            # column VIEW of the KV buffer
            acc = acc * alpha[:, None]
            acc = tle.gpu.wgmma(
                p_smem.slot(p_slot),
                tle_subslice(
                    kv.slot(buf), [0, HALF_HEAD_DIM], [BLOCK_TOKENS, HALF_HEAD_DIM]
                ),
                acc,
            )
            acc = tle.gpu.wgmma_wait(0, acc)
            tle.gpu.barrier_arrive(p_empty[p_slot], phaseIdx=p_cycle)
            p_cnt += 1
            tle.gpu.barrier_arrive(kv_empty[buf], phaseIdx=cycle)
            kv_cnt += 1

        # epilogue: consumer0 owns lse/factor; read the factor from smem
        tle.gpu.barrier_wait(o_scale_full[0], phaseIdx=seg_idx)
        factor = tl.load(tle.gpu.local_ptr(o_scale_smem.slot(0)))
        tle.gpu.barrier_arrive(o_scale_empty[0], phaseIdx=seg_idx)

        out = acc * factor[:, None]
        if seg_len == n_valid:
            # stage O[256:512) into the free Q smem hi half and signal
            # o_ready; consumer0 TMA-stores the full [64, 512]
            tl.store(
                tle.gpu.local_ptr(
                    tle_subslice(
                        q.slot(0), [0, HALF_HEAD_DIM], [BLOCK_HEADS, HALF_HEAD_DIM]
                    )
                ),
                out.to(tl.bfloat16),
            )
        else:
            part_row = (pid * 2 + tl.where(seg_idx == 0, 0, 1)) * BLOCK_HEADS
            o_base = (
                o_part_ptr
                + (part_row + head_offs[:, None]).to(tl.int64) * FULL_HEAD_DIM
            )
            tl.store(o_base + HALF_HEAD_DIM + dim_offs[None, :], out)
        tle.gpu.barrier_arrive(o_ready[0], phaseIdx=seg_idx)

        blk += seg_len
        bh_idx += 1
        blk_in_bh = 0
        seg_idx += 1


# ============================================================================
# Top-level kernels
# ============================================================================


@triton.jit
def _sparse_decode_model1_tle_kernel(
    desc_q,
    desc_o,
    o_ptr,
    main_fp8,
    main_scale_i64,
    main_bf16,
    indices_ptr,
    extra_fp8,
    extra_scale_i64,
    extra_bf16,
    extra_indices_ptr,
    attn_sink_ptr,
    topk_len_ptr,
    extra_topk_len_ptr,
    sched_ptr,
    lse_ptr,
    o_part_ptr,
    lse_part_ptr,
    sm_scale,
    TOPK,
    EXTRA_TOPK,
    NUM_MAIN_BLOCKS,
    NUM_BLOCKS,
    BLOCKS_PER_CTA,
    NUM_LONG_CTAS,
    NUM_BH,
    NUM_CTAS,
    main_stride_page,
    main_page_size,
    main_num_tokens,
    extra_stride_page,
    extra_page_size,
    extra_num_tokens,
    stride_indices_batch,
    stride_indices_seq,
    stride_extra_indices_batch,
    stride_extra_indices_seq,
    stride_lse_batch,
    stride_lse_head,
    NUM_Q_HEADS: tl.constexpr,
    SEQ_Q: tl.constexpr,
    HAVE_SINK: tl.constexpr,
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
    NUM_KV_BUFS: tl.constexpr,
    NUM_P_BUFS: tl.constexpr,
    DYNAMIC_SCHED: tl.constexpr,
    C0_REGS: tl.constexpr,
    C1_REGS: tl.constexpr,
):
    # smem + barriers. Keep this alloc order (q, kv, p, alpha, o_scale, mask):
    # it fixes every buffer's smem offset and was tuned with it.
    q = tle.gpu.alloc(
        [1, BLOCK_HEADS, FULL_HEAD_DIM],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    kv = tle.gpu.alloc(
        [NUM_KV_BUFS, BLOCK_TOKENS, FULL_HEAD_DIM],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    p_smem = tle.gpu.alloc(
        [NUM_P_BUFS, BLOCK_HEADS, BLOCK_TOKENS],
        dtype=tl.bfloat16,
        layout=None,
        scope=tle.gpu.smem,
    )
    alpha_smem = tle.gpu.alloc(
        [NUM_P_BUFS, BLOCK_HEADS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    o_scale_smem = tle.gpu.alloc(
        [1, BLOCK_HEADS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # per-block QK additive mask, produced alongside the KV buffer and sharing
    # its kv_full/kv_empty lifecycle
    mask_smem = tle.gpu.alloc(
        [NUM_KV_BUFS, BLOCK_TOKENS],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    q_empty = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1, init=tle.gpu.READY)
    q_full = tle.gpu.alloc_barriers(
        num_barriers=1,
        arrive_count=1,
        expect_bytes=BLOCK_HEADS * FULL_HEAD_DIM * 2,
    )
    kv_full = tle.gpu.alloc_barriers(num_barriers=NUM_KV_BUFS, arrive_count=1)
    # one KV buffer shared by both consumers, so it is freed only after both
    # PV reads land (consumer0 lo-half + consumer1 hi-half)
    kv_empty = tle.gpu.alloc_barriers(
        num_barriers=NUM_KV_BUFS, arrive_count=2, init=tle.gpu.READY
    )
    p_full = tle.gpu.alloc_barriers(num_barriers=NUM_P_BUFS, arrive_count=1)
    p_empty = tle.gpu.alloc_barriers(
        num_barriers=NUM_P_BUFS, arrive_count=1, init=tle.gpu.READY
    )
    o_scale_full = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1)
    o_scale_empty = tle.gpu.alloc_barriers(
        num_barriers=1, arrive_count=1, init=tle.gpu.READY
    )
    # consumer1 signals "my O half is staged in q" so consumer0 can TMA-store q
    o_ready = tle.gpu.alloc_barriers(num_barriers=1, arrive_count=1)

    # bundles for the producer: the default partition is inlined, so unlike the
    # consumers its argument grouping has no codegen consequences
    cfg = Config(
        NUM_BLOCKS,
        NUM_MAIN_BLOCKS,
        BLOCKS_PER_CTA,
        NUM_LONG_CTAS,
        NUM_BH,
        NUM_Q_HEADS=NUM_Q_HEADS,
        SEQ_Q=SEQ_Q,
        NUM_KV_BUFS=NUM_KV_BUFS,
        DYNAMIC_SCHED=DYNAMIC_SCHED,
    )
    main_cache = Cache(
        main_fp8,
        main_scale_i64,
        main_bf16,
        indices_ptr,
        topk_len_ptr,
        TOPK,
        NUM_MAIN_BLOCKS,
        main_stride_page,
        main_page_size,
        main_num_tokens,
        stride_indices_batch,
        stride_indices_seq,
        MIN_VALID_BLOCKS=1,
        HAVE_LENGTH=HAVE_TOPK_LEN,
    )
    extra_cache = Cache(
        extra_fp8,
        extra_scale_i64,
        extra_bf16,
        extra_indices_ptr,
        extra_topk_len_ptr,
        EXTRA_TOPK,
        NUM_BLOCKS - NUM_MAIN_BLOCKS,
        extra_stride_page,
        extra_page_size,
        extra_num_tokens,
        stride_extra_indices_batch,
        stride_extra_indices_seq,
        MIN_VALID_BLOCKS=0,
        HAVE_LENGTH=HAVE_EXTRA and HAVE_EXTRA_TOPK_LEN,
    )
    ch = Channel(
        q,
        kv,
        mask_smem,
        p_smem,
        alpha_smem,
        o_scale_smem,
        q_empty,
        q_full,
        kv_full,
        kv_empty,
        p_full,
        p_empty,
        o_scale_full,
        o_scale_empty,
        o_ready,
    )

    tle.gpu.warp_specialize(
        [
            (
                _model1_producer,
                (desc_q, sched_ptr, cfg, ch, main_cache, extra_cache),
            ),
            (
                _model1_consumer0,
                (
                    desc_o,
                    o_ptr,
                    attn_sink_ptr,
                    topk_len_ptr,
                    extra_topk_len_ptr,
                    sched_ptr,
                    lse_ptr,
                    o_part_ptr,
                    lse_part_ptr,
                    sm_scale,
                    TOPK,
                    EXTRA_TOPK,
                    NUM_MAIN_BLOCKS,
                    NUM_BLOCKS,
                    BLOCKS_PER_CTA,
                    NUM_LONG_CTAS,
                    NUM_BH,
                    stride_lse_batch,
                    stride_lse_head,
                    q,
                    kv,
                    mask_smem,
                    p_smem,
                    alpha_smem,
                    o_scale_smem,
                    q_empty,
                    q_full,
                    kv_full,
                    kv_empty,
                    p_full,
                    p_empty,
                    o_scale_full,
                    o_scale_empty,
                    o_ready,
                    NUM_Q_HEADS,
                    SEQ_Q,
                    HAVE_SINK,
                    HAVE_TOPK_LEN,
                    HAVE_EXTRA,
                    HAVE_EXTRA_TOPK_LEN,
                    NUM_KV_BUFS,
                    NUM_P_BUFS,
                    DYNAMIC_SCHED,
                    1,
                ),
            ),
            (
                _model1_consumer1,
                (
                    desc_o,
                    o_ptr,
                    topk_len_ptr,
                    extra_topk_len_ptr,
                    sched_ptr,
                    o_part_ptr,
                    TOPK,
                    EXTRA_TOPK,
                    NUM_MAIN_BLOCKS,
                    NUM_BLOCKS,
                    BLOCKS_PER_CTA,
                    NUM_LONG_CTAS,
                    NUM_BH,
                    q,
                    kv,
                    p_smem,
                    alpha_smem,
                    o_scale_smem,
                    kv_full,
                    kv_empty,
                    p_full,
                    p_empty,
                    o_scale_full,
                    o_scale_empty,
                    o_ready,
                    NUM_Q_HEADS,
                    SEQ_Q,
                    HAVE_TOPK_LEN,
                    HAVE_EXTRA,
                    HAVE_EXTRA_TOPK_LEN,
                    NUM_KV_BUFS,
                    NUM_P_BUFS,
                    DYNAMIC_SCHED,
                    2,
                ),
            ),
        ],
        [4, 4],
        [C0_REGS, C1_REGS],
    )


@triton.jit
def _valid_blocks_tile(
    topk_len_ptr,
    extra_topk_len_ptr,
    bh_offs,
    bh_mask,
    BH_PER_BATCH,
    NUM_MAIN_BLOCKS,
    NUM_EXTRA_BLOCKS,
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
):
    """Vectorized twin of _seg_valid_blocks for the scheduler: per-batch-head
    valid block count over a [CHUNK] tile of bh indices (0 outside bh_mask)."""
    batch_idx = bh_offs // BH_PER_BATCH
    if HAVE_TOPK_LEN:
        main_len = tl.load(topk_len_ptr + batch_idx, mask=bh_mask, other=BLOCK_TOKENS)
        n_main = tl.minimum(
            tl.maximum((main_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS, 1),
            NUM_MAIN_BLOCKS,
        )
    else:
        n_main = tl.zeros_like(bh_offs) + NUM_MAIN_BLOCKS
    if HAVE_EXTRA_TOPK_LEN:
        extra_len = tl.load(extra_topk_len_ptr + batch_idx, mask=bh_mask, other=0)
        n_extra = tl.minimum(
            tl.maximum((extra_len + BLOCK_TOKENS - 1) // BLOCK_TOKENS, 0),
            NUM_EXTRA_BLOCKS,
        )
    else:
        n_extra = tl.zeros_like(bh_offs) + NUM_EXTRA_BLOCKS
    return tl.where(bh_mask, n_main + n_extra, 0)


@triton.jit
def _model1_sched_kernel(
    topk_len_ptr,
    extra_topk_len_ptr,
    sched_ptr,  # int32 [NUM_BH+3+2P]: prefix[NUM_BH], total, blocks_per_cta,
    # num_long_ctas, start_bh[P], start_blk_in_bh[P]
    TOPK,
    EXTRA_TOPK,
    NUM_MAIN_BLOCKS,
    NUM_EXTRA_BLOCKS,
    NUM_BH,
    NUM_CTAS,
    BH_PER_BATCH,  # bh items per batch element = SEQ_Q * NUM_HEAD_BLOCKS
    NUM_CHUNKS: tl.constexpr,
    CHUNK: tl.constexpr,
    BLOCK_CTAS: tl.constexpr,  # next_pow2(NUM_CTAS)
    HAVE_TOPK_LEN: tl.constexpr,
    HAVE_EXTRA_TOPK_LEN: tl.constexpr,
):
    """Balanced splitkv schedule over VALID blocks (single CTA, launched once
    before the main kernel; cudagraph-safe). Mirrors FlashMLA's length-aware
    tile scheduler: work is split evenly by the number of blocks that actually
    carry data, so CTAs don't idle on batch-heads with short dynamic lengths."""
    cta_offs = tl.arange(0, BLOCK_CTAS)

    # pass 1: total valid blocks
    total = 0
    for chunk_start in range(0, NUM_CHUNKS * CHUNK, CHUNK):
        bh_offs = chunk_start + tl.arange(0, CHUNK)
        valid_counts = _valid_blocks_tile(
            topk_len_ptr,
            extra_topk_len_ptr,
            bh_offs,
            bh_offs < NUM_BH,
            BH_PER_BATCH,
            NUM_MAIN_BLOCKS,
            NUM_EXTRA_BLOCKS,
            HAVE_TOPK_LEN,
            HAVE_EXTRA_TOPK_LEN,
        )
        total += tl.sum(valid_counts, 0)

    blocks_per_cta = total // NUM_CTAS
    num_long_ctas = total - blocks_per_cta * NUM_CTAS
    tl.store(sched_ptr + NUM_BH, total)
    tl.store(sched_ptr + NUM_BH + 1, blocks_per_cta)
    tl.store(sched_ptr + NUM_BH + 2, num_long_ctas)

    # pass 2: exclusive prefix per bh + per-CTA start (bh_idx, blk_in_bh).
    # start_bh(p) counts bhs whose inclusive prefix <= cta_start(p), and
    # start_prefix tracks the largest such prefix, so
    # blk_in_bh = cta_start - start_prefix.
    cta_starts = cta_offs * blocks_per_cta + tl.minimum(cta_offs, num_long_ctas)
    start_bh = tl.zeros([BLOCK_CTAS], tl.int32)
    start_prefix = tl.zeros([BLOCK_CTAS], tl.int32)
    running = 0
    for chunk_start in range(0, NUM_CHUNKS * CHUNK, CHUNK):
        bh_offs = chunk_start + tl.arange(0, CHUNK)
        bh_mask = bh_offs < NUM_BH
        valid_counts = _valid_blocks_tile(
            topk_len_ptr,
            extra_topk_len_ptr,
            bh_offs,
            bh_mask,
            BH_PER_BATCH,
            NUM_MAIN_BLOCKS,
            NUM_EXTRA_BLOCKS,
            HAVE_TOPK_LEN,
            HAVE_EXTRA_TOPK_LEN,
        )
        prefix_incl = tl.cumsum(valid_counts, 0) + running
        tl.store(sched_ptr + bh_offs, prefix_incl - valid_counts, mask=bh_mask)
        before_start = (prefix_incl[None, :] <= cta_starts[:, None]) & bh_mask[None, :]
        start_bh += tl.sum(before_start.to(tl.int32), 1)
        start_prefix = tl.maximum(
            start_prefix, tl.max(tl.where(before_start, prefix_incl[None, :], 0), 1)
        )
        running += tl.sum(valid_counts, 0)

    cta_mask = cta_offs < NUM_CTAS
    tl.store(sched_ptr + NUM_BH + 3 + cta_offs, start_bh, mask=cta_mask)
    tl.store(
        sched_ptr + NUM_BH + 3 + NUM_CTAS + cta_offs,
        cta_starts - start_prefix,
        mask=cta_mask,
    )


@triton.jit
def _splitkv_combine_kernel(
    o_part_ptr,  # [P, 2, 64, 512] f32, per-partial-segment normalized O
    lse_part_ptr,  # [P, 2, 64]      f32, natural-log LSE
    attn_sink_ptr,  # [NUM_Q_HEADS]   f32
    o_ptr,  # [B, SEQ_Q, NUM_Q_HEADS, 512] bf16, contiguous
    lse_ptr,  # [B, NUM_Q_HEADS, SEQ_Q]      f32,  contiguous
    sched_ptr,
    NUM_Q_HEADS,
    SEQ_Q,
    NUM_HEAD_BLOCKS,
    NUM_BLOCKS,
    BLOCKS_PER_CTA,
    NUM_LONG_CTAS,
    NUM_BH,
    BLOCK_SPLITS: tl.constexpr,  # next_pow2(P), covers the max split count
    HAVE_SINK: tl.constexpr,
    DYNAMIC_SCHED: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_idx = pid // (SEQ_Q * NUM_Q_HEADS)
    rest = pid % (SEQ_Q * NUM_Q_HEADS)
    seq_q_idx = rest // NUM_Q_HEADS
    head_idx = rest % NUM_Q_HEADS
    head_block_idx = head_idx // BLOCK_HEADS
    head_in_block = head_idx % BLOCK_HEADS
    bh_idx = (batch_idx * SEQ_Q + seq_q_idx) * NUM_HEAD_BLOCKS + head_block_idx

    # valid-block range of this batch-head plus the schedule split
    if DYNAMIC_SCHED:
        prefix0 = tl.load(sched_ptr + bh_idx)
        prefix1 = tl.load(sched_ptr + bh_idx + 1)
        blocks_per_cta = tl.load(sched_ptr + NUM_BH + 1)
        num_long_ctas = tl.load(sched_ptr + NUM_BH + 2)
    else:
        prefix0 = bh_idx * NUM_BLOCKS
        prefix1 = prefix0 + NUM_BLOCKS
        blocks_per_cta = BLOCKS_PER_CTA
        num_long_ctas = NUM_LONG_CTAS
    cta_first = _cta_of_block(prefix0, blocks_per_cta, num_long_ctas)
    cta_last = _cta_of_block(prefix1 - 1, blocks_per_cta, num_long_ctas)
    num_splits = cta_last - cta_first + 1
    if num_splits <= 1:
        return  # a single CTA covered the whole bh and wrote out/lse directly

    # the partial of CTA c for this bh sits at slot 0 if the bh is c's FIRST
    # segment (c's range starts inside the bh), else at slot 1 (its tail)
    split_offs = tl.arange(0, BLOCK_SPLITS)
    cta_ids = cta_first + split_offs
    split_mask = split_offs < num_splits
    cta_start = cta_ids * blocks_per_cta + tl.minimum(cta_ids, num_long_ctas)
    part_rows = (
        cta_ids * 2 + tl.where(cta_start >= prefix0, 0, 1)
    ) * BLOCK_HEADS + head_in_block
    lse_parts = tl.load(lse_part_ptr + part_rows, mask=split_mask, other=float("-inf"))
    max_lse = tl.max(lse_parts, 0)
    max_safe = tl.where(max_lse == float("-inf"), 0.0, max_lse)
    sum_weights = tl.sum(tl.where(split_mask, tl.exp(lse_parts - max_safe), 0.0), 0)

    empty = sum_weights == 0.0
    lse_global = max_safe + tl.log(sum_weights)  # natural log, excludes sink
    lse_safe = tl.where(empty, 0.0, lse_global)
    tl.store(
        lse_ptr + batch_idx * NUM_Q_HEADS * SEQ_Q + head_idx * SEQ_Q + seq_q_idx,
        tl.where(empty, float("inf"), lse_global),
    )

    dim_offs = tl.arange(0, FULL_HEAD_DIM)
    acc = tl.zeros([FULL_HEAD_DIM], tl.float32)
    for s in range(num_splits):
        cta = cta_first + s
        cta_s = cta * blocks_per_cta + tl.minimum(cta, num_long_ctas)
        row = (cta * 2 + tl.where(cta_s >= prefix0, 0, 1)) * BLOCK_HEADS + head_in_block
        weight = tl.exp(tl.load(lse_part_ptr + row) - lse_safe)
        o_partial = tl.load(o_part_ptr + row.to(tl.int64) * FULL_HEAD_DIM + dim_offs)
        acc += weight * o_partial

    if HAVE_SINK:
        sink = tl.load(attn_sink_ptr + head_idx)
        acc = acc / (1.0 + tl.exp(sink - lse_safe))
    acc = tl.where(empty, 0.0, acc)

    tl.store(
        o_ptr
        + (
            ((batch_idx * SEQ_Q + seq_q_idx) * NUM_Q_HEADS + head_idx) * FULL_HEAD_DIM
        ).to(tl.int64)
        + dim_offs,
        acc.to(tl.bfloat16),
    )


# ============================================================================
# Host entry points
# ============================================================================


def _cache_views(cache: torch.Tensor):
    """Flat fp8 / packed-scale-i64 / bf16 views over a MODEL1 paged cache.
    Returns (fp8, scale_i64, bf16, stride_page, page_size, num_tokens)."""
    assert _cache_is_supported(cache), (
        "MODEL1 TLE cache must be 584 B/token with 16B-aligned pages; "
        "check can_use_model1_tle() before calling"
    )
    num_pages, page_size = cache.shape[0], cache.shape[1]
    u8 = cache.view(torch.uint8)
    stride_page = u8.stride(0)
    nbytes = u8.untyped_storage().nbytes() - u8.storage_offset()
    flat_u8 = u8.as_strided((nbytes & ~7,), (1,))
    return (
        flat_u8.view(torch.float8_e4m3fn),
        flat_u8.view(torch.int64),
        flat_u8.view(torch.bfloat16),
        stride_page,
        page_size,
        num_pages * page_size,
    )


def _cache_is_supported(cache: torch.Tensor) -> bool:
    """The 128-bit gathers require 16B-aligned pages, same implicit contract as
    the FlashMLA CUDA kernel."""
    if cache.dtype not in (torch.uint8, torch.float8_e4m3fn):
        return False
    if cache.shape[-1] != PAGE_TOKEN_BYTES:
        return False
    u8 = cache.view(torch.uint8)
    return (
        u8.stride(-1) == 1 and u8.stride(0) % 16 == 0 and u8.storage_offset() % 16 == 0
    )


def can_use_model1_tle(
    q: torch.Tensor,
    kv_584: torch.Tensor,
    indices: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    extra_kv: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
) -> bool:
    """Whether the warp-specialized MODEL1 kernel can serve this call. Callers
    fall back to the portable MODEL1 kernel when this is False."""
    major, _ = get_device_capability()
    if major != 9:  # wgmma + TMA warp specialization is sm90-only
        return False
    if q.dtype != torch.bfloat16 or q.shape[-1] != FULL_HEAD_DIM.value:
        return False
    if q.shape[2] % BLOCK_HEADS.value != 0:
        return False
    if indices.shape[-1] % BLOCK_TOKENS.value != 0:
        return False
    # the kernel writes O and LSE through flat views / a TMA descriptor
    if not (out.is_contiguous() and lse.is_contiguous()):
        return False
    if not _cache_is_supported(kv_584):
        return False
    if extra_kv is not None:
        if not _cache_is_supported(extra_kv):
            return False
        if extra_indices is None or extra_indices.shape[-1] % BLOCK_TOKENS.value != 0:
            return False
    return True


def sparse_decode_model1_tle(
    q: torch.Tensor,  # [B, SQ, HQ, 512] bf16
    kv_584: torch.Tensor,  # [num_pages, page_size, 1, 584] u8/fp8
    indices: torch.Tensor,  # [B, SQ, TOPK] int32
    attn_sink: torch.Tensor | None = None,  # [HQ] f32
    topk_length: torch.Tensor | None = None,  # [B] int32
    extra_kv: torch.Tensor | None = None,  # second MODEL1 cache
    extra_indices: torch.Tensor | None = None,  # [B, SQ, EXTRA_TOPK] int32
    extra_topk_length: torch.Tensor | None = None,  # [B] int32
    out: torch.Tensor | None = None,  # [B, SQ, HQ, 512] bf16
    lse: torch.Tensor | None = None,  # [B, HQ, SQ] f32
    sm_scale: float | None = None,
    num_ctas: int | None = None,  # persistent grid size; None = SM count
    num_bufs: int = 2,  # KV smem buffers
    c0_regs: int = DEFAULT_C0_REGS,
    c1_regs: int = DEFAULT_C1_REGS,
):
    batch, seq_q, num_q_heads, d_qk = q.shape
    assert d_qk == FULL_HEAD_DIM.value and num_q_heads % BLOCK_HEADS.value == 0
    if not q.is_contiguous():
        q = q.contiguous()
    topk = indices.shape[-1]
    assert topk % BLOCK_TOKENS.value == 0
    if indices.stride(-1) != 1:
        indices = indices.contiguous()

    triton.set_allocator(_alloc_fn)

    (
        main_fp8,
        main_scale_i64,
        main_bf16,
        main_stride_page,
        main_page_size,
        main_num_tokens,
    ) = _cache_views(kv_584)

    have_extra = extra_kv is not None
    if have_extra:
        assert extra_indices is not None
        extra_topk = extra_indices.shape[-1]
        assert extra_topk % BLOCK_TOKENS.value == 0
        if extra_indices.stride(-1) != 1:
            extra_indices = extra_indices.contiguous()
        (
            extra_fp8,
            extra_scale_i64,
            extra_bf16,
            extra_stride_page,
            extra_page_size,
            extra_num_tokens,
        ) = _cache_views(extra_kv)
    else:
        assert extra_indices is None and extra_topk_length is None
        extra_topk = 0
        extra_fp8, extra_scale_i64, extra_bf16 = main_fp8, main_scale_i64, main_bf16
        extra_stride_page, extra_page_size = main_stride_page, main_page_size
        extra_num_tokens = 0
        extra_indices = indices

    if out is None:
        out = torch.empty(
            batch,
            seq_q,
            num_q_heads,
            FULL_HEAD_DIM.value,
            dtype=q.dtype,
            device=q.device,
        )
    if lse is None:
        lse = torch.empty(
            batch, num_q_heads, seq_q, dtype=torch.float32, device=q.device
        )
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5
    assert out.is_contiguous() and lse.is_contiguous()

    q_flat = q.view(batch * seq_q * num_q_heads, d_qk)
    out_flat = out.view(batch * seq_q * num_q_heads, FULL_HEAD_DIM.value)
    desc_q = TensorDescriptor(
        q_flat,
        shape=[batch * seq_q * num_q_heads, d_qk],
        strides=[d_qk, 1],
        block_shape=[BLOCK_HEADS.value, FULL_HEAD_DIM.value],
    )
    # the consumers stage O into the free Q smem buffer, then TMA it out
    desc_o = TensorDescriptor(
        out_flat,
        shape=[batch * seq_q * num_q_heads, FULL_HEAD_DIM.value],
        strides=[FULL_HEAD_DIM.value, 1],
        block_shape=[BLOCK_HEADS.value, FULL_HEAD_DIM.value],
    )

    # persistent + splitkv scheduling. With dynamic lengths the schedule is
    # computed on device over the blocks that actually carry data; otherwise
    # the static formula is used and no metadata kernel runs.
    num_head_blocks = num_q_heads // BLOCK_HEADS.value
    num_main_blocks = topk // BLOCK_TOKENS.value
    num_blocks = num_main_blocks + extra_topk // BLOCK_TOKENS.value
    num_bh = batch * seq_q * num_head_blocks
    total_blocks = num_bh * num_blocks
    num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    grid_ctas = min(num_ctas or num_sms, total_blocks)
    blocks_per_cta, num_long_ctas = (
        total_blocks // grid_ctas,
        total_blocks % grid_ctas,
    )
    # The dynamic schedule costs a metadata kernel and forces the combine
    # kernel to run, which only pays off once a batch-head holds enough blocks
    # for length-driven imbalance to matter. Below that, walk the padded space
    # and let the per-token mask drop out-of-range tokens instead.
    has_length = topk_length is not None or extra_topk_length is not None
    dynamic_sched = has_length and num_blocks >= MIN_BLOCKS_FOR_DYNAMIC_SCHED

    # a CTA writes at most two partial segments: its first (slot 0) and its
    # last (slot 1); middle segments always cover a whole bh and write direct
    o_part = torch.empty(
        grid_ctas,
        2,
        BLOCK_HEADS.value,
        FULL_HEAD_DIM.value,
        dtype=torch.float32,
        device=q.device,
    )
    lse_part = torch.empty(
        grid_ctas, 2, BLOCK_HEADS.value, dtype=torch.float32, device=q.device
    )

    if dynamic_sched:
        sched = torch.empty(
            num_bh + 3 + 2 * grid_ctas, dtype=torch.int32, device=q.device
        )
        CHUNK = 128
        _model1_sched_kernel[(1,)](
            topk_length if topk_length is not None else indices,  # dummy
            extra_topk_length if extra_topk_length is not None else indices,  # dummy
            sched,
            topk,
            extra_topk,
            num_main_blocks,
            num_blocks - num_main_blocks,
            num_bh,
            grid_ctas,
            seq_q * num_head_blocks,
            NUM_CHUNKS=triton.cdiv(num_bh, CHUNK),
            CHUNK=CHUNK,
            BLOCK_CTAS=triton.next_power_of_2(grid_ctas),
            HAVE_TOPK_LEN=topk_length is not None,
            HAVE_EXTRA_TOPK_LEN=extra_topk_length is not None,
            num_warps=4,
        )
    else:
        sched = indices  # dummy, never read

    _sparse_decode_model1_tle_kernel[(grid_ctas,)](
        desc_q,
        desc_o,
        out_flat,
        main_fp8,
        main_scale_i64,
        main_bf16,
        indices,
        extra_fp8,
        extra_scale_i64,
        extra_bf16,
        extra_indices,
        attn_sink if attn_sink is not None else lse,  # dummy when no sink
        topk_length if topk_length is not None else indices,  # dummy
        extra_topk_length if extra_topk_length is not None else indices,  # dummy
        sched,
        lse,
        o_part,
        lse_part,
        sm_scale,
        topk,
        extra_topk,
        num_main_blocks,
        num_blocks,
        blocks_per_cta,
        num_long_ctas,
        num_bh,
        grid_ctas,
        main_stride_page,
        main_page_size,
        main_num_tokens,
        extra_stride_page,
        extra_page_size,
        extra_num_tokens,
        indices.stride(0),
        indices.stride(1),
        extra_indices.stride(0),
        extra_indices.stride(1),
        lse.stride(0),
        lse.stride(1),
        NUM_Q_HEADS=num_q_heads,
        SEQ_Q=seq_q,
        HAVE_SINK=attn_sink is not None,
        HAVE_TOPK_LEN=topk_length is not None,
        HAVE_EXTRA=have_extra,
        HAVE_EXTRA_TOPK_LEN=extra_topk_length is not None,
        NUM_KV_BUFS=num_bufs,
        NUM_P_BUFS=2,
        DYNAMIC_SCHED=dynamic_sched,
        C0_REGS=c0_regs,
        C1_REGS=c1_regs,
        num_warps=4,
    )

    all_direct = not dynamic_sched and (
        num_blocks == 1 or (num_long_ctas == 0 and blocks_per_cta % num_blocks == 0)
    )
    if not all_direct:
        _splitkv_combine_kernel[(batch * seq_q * num_q_heads,)](
            o_part,
            lse_part,
            attn_sink if attn_sink is not None else lse,  # dummy when no sink
            out,
            lse,
            sched,
            num_q_heads,
            seq_q,
            num_head_blocks,
            num_blocks,
            blocks_per_cta,
            num_long_ctas,
            num_bh,
            BLOCK_SPLITS=triton.next_power_of_2(max(grid_ctas, 2)),
            HAVE_SINK=attn_sink is not None,
            DYNAMIC_SCHED=dynamic_sched,
        )

    return out, lse
