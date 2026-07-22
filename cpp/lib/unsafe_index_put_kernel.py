"""
_unsafe_index_put Triton kernel — standalone, 供 C++ wrapper 通过 TritonJITFunction 调用。

与 Python 版本的区别:
- 直接 @triton.jit, 无 @libentry() 包装 (C++ 端自己做 dispatch)
- BLOCK 不作为 constexpr (C++ 端传入)
- 输出通过 out_ptr 写入 (C++ 端预先分配并用 DMA copy 初始化)
"""

import triton
import triton.language as tl


@triton.jit
def unsafe_index_put_kernel_cpp(
    out_ptr,
    values_ptr,
    idx0_ptr,
    idx1_ptr,
    idx2_ptr,
    idx3_ptr,
    idx_div0,
    idx_div1,
    idx_div2,
    idx_div3,
    ts_0_0,
    ts_0_1,
    ts_0_2,
    ts_0_3,
    ts_1_0,
    ts_1_1,
    ts_1_2,
    ts_1_3,
    ts_2_0,
    ts_2_1,
    ts_2_2,
    ts_2_3,
    ts_3_0,
    ts_3_1,
    ts_3_2,
    ts_3_3,
    val_adv0,
    val_adv1,
    val_adv2,
    val_adv3,
    self_adv_stride0,
    self_adv_stride1,
    self_adv_stride2,
    self_adv_stride3,
    self_adv_size0,
    self_adv_size1,
    self_adv_size2,
    self_adv_size3,
    suf_div0,
    suf_div1,
    suf_div2,
    suf_div3,
    self_suf_stride0,
    self_suf_stride1,
    self_suf_stride2,
    self_suf_stride3,
    val_suf_stride0,
    val_suf_stride1,
    val_suf_stride2,
    val_suf_stride3,
    idx_numel,
    suffix_numel,
    N,
    M: tl.constexpr,
    IDX_NDIM: tl.constexpr,
    SUF_NDIM: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pid < N

    # 转换索引指针为 pointer type
    if M >= 1:
        idx0_ptr = idx0_ptr.to(tl.pointer_type(tl.int64))
    if M >= 2:
        idx1_ptr = idx1_ptr.to(tl.pointer_type(tl.int64))
    if M >= 3:
        idx2_ptr = idx2_ptr.to(tl.pointer_type(tl.int64))
    if M >= 4:
        idx3_ptr = idx3_ptr.to(tl.pointer_type(tl.int64))

    idx_pos = pid // suffix_numel
    suf_pos = pid % suffix_numel

    self_off = tl.zeros((BLOCK,), dtype=tl.int64)
    val_off = tl.zeros((BLOCK,), dtype=tl.int64)

    toff_0 = tl.zeros((BLOCK,), dtype=tl.int64)
    toff_1 = tl.zeros((BLOCK,), dtype=tl.int64)
    toff_2 = tl.zeros((BLOCK,), dtype=tl.int64)
    toff_3 = tl.zeros((BLOCK,), dtype=tl.int64)

    rem_idx = idx_pos

    if IDX_NDIM >= 1:
        coord = rem_idx // idx_div0
        rem_idx = rem_idx % idx_div0
        val_off += coord * val_adv0
        if M >= 1:
            toff_0 += coord * ts_0_0
        if M >= 2:
            toff_1 += coord * ts_1_0
        if M >= 3:
            toff_2 += coord * ts_2_0
        if M >= 4:
            toff_3 += coord * ts_3_0

    if IDX_NDIM >= 2:
        coord = rem_idx // idx_div1
        rem_idx = rem_idx % idx_div1
        val_off += coord * val_adv1
        if M >= 1:
            toff_0 += coord * ts_0_1
        if M >= 2:
            toff_1 += coord * ts_1_1
        if M >= 3:
            toff_2 += coord * ts_2_1
        if M >= 4:
            toff_3 += coord * ts_3_1

    if IDX_NDIM >= 3:
        coord = rem_idx // idx_div2
        rem_idx = rem_idx % idx_div2
        val_off += coord * val_adv2
        if M >= 1:
            toff_0 += coord * ts_0_2
        if M >= 2:
            toff_1 += coord * ts_1_2
        if M >= 3:
            toff_2 += coord * ts_2_2
        if M >= 4:
            toff_3 += coord * ts_3_2

    if IDX_NDIM >= 4:
        coord = rem_idx // idx_div3
        rem_idx = rem_idx % idx_div3
        val_off += coord * val_adv3
        if M >= 1:
            toff_0 += coord * ts_0_3
        if M >= 2:
            toff_1 += coord * ts_1_3
        if M >= 3:
            toff_2 += coord * ts_2_3
        if M >= 4:
            toff_3 += coord * ts_3_3

    if M >= 1:
        ind = tl.load(idx0_ptr + toff_0, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size0, ind)
        self_off += ind * self_adv_stride0
    if M >= 2:
        ind = tl.load(idx1_ptr + toff_1, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size1, ind)
        self_off += ind * self_adv_stride1
    if M >= 3:
        ind = tl.load(idx2_ptr + toff_2, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size2, ind)
        self_off += ind * self_adv_stride2
    if M >= 4:
        ind = tl.load(idx3_ptr + toff_3, mask=mask, other=0)
        ind = ind.to(tl.int64)
        ind = tl.where(ind < 0, ind + self_adv_size3, ind)
        self_off += ind * self_adv_stride3

    rem_suf = suf_pos
    if SUF_NDIM >= 1:
        coord = rem_suf // suf_div0
        rem_suf = rem_suf % suf_div0
        self_off += coord * self_suf_stride0
        val_off += coord * val_suf_stride0
    if SUF_NDIM >= 2:
        coord = rem_suf // suf_div1
        rem_suf = rem_suf % suf_div1
        self_off += coord * self_suf_stride1
        val_off += coord * val_suf_stride1
    if SUF_NDIM >= 3:
        coord = rem_suf // suf_div2
        rem_suf = rem_suf % suf_div2
        self_off += coord * self_suf_stride2
        val_off += coord * val_suf_stride2
    if SUF_NDIM >= 4:
        coord = rem_suf // suf_div3
        rem_suf = rem_suf % suf_div3
        self_off += coord * self_suf_stride3
        val_off += coord * val_suf_stride3

    v = tl.load(values_ptr + val_off, mask=mask, other=0.0)
    if ACCUMULATE:
        tl.atomic_add(out_ptr + self_off, v, mask=mask)
    else:
        tl.store(out_ptr + self_off, v, mask=mask)
