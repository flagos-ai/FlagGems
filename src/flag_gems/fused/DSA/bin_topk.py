from typing import Optional

import torch
import triton
import triton.language as tl
import numpy as np

@triton.jit
def convert_to_uint16(x):
    hval = x.cast(dtype=tl.float16)
    bits_uint = hval.cast(dtype=tl.uint16, bitcast=True) # 相当于reinterpret
    bits_uint = tl.where(x<0, ~bits_uint & (0xFFFF), bits_uint | (0x8000))
    return bits_uint >> 8

@triton.jit
def convert_to_uint32(x):
    bits_uint = x.cast(dtype=tl.uint32, bitcast=True)
    bits_uint = tl.where(x<0, ~bits_uint & tl.cast((0xFFFFFFFF), tl.uint32), bits_uint | tl.cast((0x80000000), tl.uint32))
    return bits_uint


@triton.autotune(
    configs=[
        triton.Config({'BS': 512, 'BK': 256}, num_stages=2, num_warps=2),
        triton.Config({'BS': 1024, 'BK': 256}, num_stages=2, num_warps=2),
        triton.Config({'BS': 2048, 'BK': 256}, num_stages=2, num_warps=4),
        triton.Config({'BS': 4096, 'BK': 512}, num_stages=3, num_warps=4),
        triton.Config({'BS': 8192, 'BK': 512}, num_stages=3, num_warps=8),
        triton.Config({'BS': 8192, 'BK': 1024}, num_stages=3, num_warps=8),
    ],
    key=['S', 'K'],
)

@triton.jit
def kernel_bucket_sort_topk( # grid(B, BS)
    inputs, # (B, S) 注意，没有 H，因为MLA基于MQA和MHA而非GQA
    indices, # (B, K) topk 索引数组
    # s_histogram,
    s_input_ids, # 下一轮中待筛选的数据索引
    starts,
    ends,
    S: tl.constexpr, # sequence length
    K: tl.constexpr, # k of topk
    HISTOGRAM_SIZE: tl.constexpr,
    SMEM_INPUT_SIZE: tl.constexpr,
    BS: tl.constexpr, # block size of S
    BK: tl.constexpr,
):
    # 获取线程块id
    i_b = tl.program_id(0)

    # 块基础指针定义
    s_base = inputs + i_b * S 
    indices_base = indices + i_b * K
    s_input_ids_base = s_input_ids + i_b * SMEM_INPUT_SIZE

    # 直方图初始化
    s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)

    # 支持变长
    l_start_idx = tl.load(starts + i_b).to(tl.int32)
    l_end_idx = tl.load(ends + i_b).to(tl.int32)

    # 记录topk数组还剩多少可以被填满
    l_new_topk = K

    TS = tl.cdiv(S, BS)
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        input = tl.load(s_base + input_idx, input_mask).to(tl.float32)
        # input = tl.rand(42, tl.arange(0, BS))
        inval_int16 = convert_to_uint16(input)
        s_histogram += inval_int16.to(tl.int32).histogram(HISTOGRAM_SIZE)

    # if i_b==0:
    #     print("s_histogram", s_histogram)  

    s_histogram = s_histogram.cumsum(0, reverse=True) # 后缀和
    
    mv_idx = tl.arange(1,HISTOGRAM_SIZE+1) % HISTOGRAM_SIZE # 构造错位索引矩阵
    # cond = (s_histogram > l_new_topk) & (s_histogram.gather(mv_idx, 0) <= l_new_topk)
    # l_threshold_bin_id = tl.where(cond, tl.arange(1, HISTOGRAM_SIZE+1), 0).max(0)
    # l_threshold_bin_id = tl.where(l_threshold_bin_id>0, l_threshold_bin_id, HISTOGRAM_SIZE) - 1
    #因为无法设置第257个桶而加的补救措施，如果没有桶被找出，则赋值为最后一个
    # 对应tilelang中的如下语句：
    #   if s_histogram[tx] > l_new_topk and s_histogram[tx + 1] <= l_new_topk:
    #   s_threshold_bin_id[0] = tx

    cond = (s_histogram > l_new_topk) & ((s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0))
    l_threshold_bin_id = cond.argmax(0)

    l_new_topk -= tl.where(tl.arange(0, HISTOGRAM_SIZE)==l_threshold_bin_id+1, s_histogram, 0).max(0)
    sum = 0
    thre_bin_sum = 0
    for s in range(TS):
        input_idx = s * BS + tl.arange(0, BS)
        input_mask = (input_idx < l_end_idx) & (input_idx >= l_start_idx) & (input_idx < S)
        input = tl.load(s_base + input_idx, input_mask).to(tl.float32)
        # input = tl.rand(42, tl.arange(0, BS))
        inval_int16 = convert_to_uint16(input)

        over_thre = inval_int16.to(tl.int32) > l_threshold_bin_id
        cur_sum = over_thre.to(tl.int32).sum(0)
        
        eq_thre = inval_int16.to(tl.int32) == l_threshold_bin_id
        thre_bin_cur_sum = eq_thre.to(tl.int32).sum(0)

        topk_idx = over_thre.to(tl.int32).cumsum(0)
        thre_bin_idx = eq_thre.to(tl.int32).cumsum(0)

        concat_mask = tl.cat(over_thre, eq_thre, True)
        concat_input = tl.cat(input_idx, input_idx, True)
        concat_pointer_matrix = tl.cat(indices_base + sum + topk_idx - 1, s_input_ids_base + thre_bin_sum + thre_bin_idx - 1, True)

        tl.store(concat_pointer_matrix, concat_input, mask = concat_mask)
            
        thre_bin_sum += thre_bin_cur_sum
        sum += cur_sum

    round = 0
    while round < 4 and l_new_topk > 0 :
        round += 1
        ss = tl.cdiv(thre_bin_sum, BK)
        s_histogram = tl.zeros([HISTOGRAM_SIZE], dtype=tl.int32)

        for s in range(ss):
            s_input_idx = s * BK + tl.arange(0, BK)
            s_input_idx_mask = s_input_idx < thre_bin_sum
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask)
            s_input = tl.load(s_base + input_idx, s_input_idx_mask).to(tl.float32)
            inval_int32 = (convert_to_uint32(s_input) >> (24-round*8)) & 0xFF # 保证除了最后八位外都为0
            s_histogram += inval_int32.to(tl.int32).histogram(HISTOGRAM_SIZE)

        s_histogram = s_histogram.cumsum(0, reverse=True) # 后缀和
        mv_idx = tl.arange(1,HISTOGRAM_SIZE+1) % HISTOGRAM_SIZE # 构造错位索引矩阵
        # cond = (s_histogram > l_new_topk) & (s_histogram.gather(mv_idx, 0) <= l_new_topk)
        # l_threshold_bin_id = tl.where(cond, tl.arange(1, HISTOGRAM_SIZE+1), 0).max(0)
        # l_threshold_bin_id = tl.where(l_threshold_bin_id>0, l_threshold_bin_id, HISTOGRAM_SIZE) - 1
        cond = (s_histogram > l_new_topk) & ((s_histogram.gather(mv_idx, 0) <= l_new_topk) | (mv_idx == 0))
        l_threshold_bin_id = cond.argmax(0)
        l_new_topk -= tl.where(tl.arange(0, HISTOGRAM_SIZE)==l_threshold_bin_id+1, s_histogram, 0).max(0)
        thre_bin_sum, old_thre_bin_sum = 0, thre_bin_sum

        for s in range(ss):
            s_input_idx = s * BK + tl.arange(0, BK)
            s_input_idx_mask = s_input_idx < old_thre_bin_sum
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask)
            s_input = tl.load(s_base + input_idx, s_input_idx_mask).to(tl.float32)            
            inval_int32 = (convert_to_uint32(s_input) >> (24-round*8)) & 0xFF # 保证除了最后八位外都为0

            over_thre = inval_int32.to(tl.int32) > l_threshold_bin_id
            cur_sum = over_thre.to(tl.int32).sum(0)
            
            eq_thre = inval_int32.to(tl.int32) == l_threshold_bin_id
            thre_bin_cur_sum = eq_thre.to(tl.int32).sum(0)

            topk_idx = over_thre.to(tl.int32).cumsum(0)
            thre_bin_idx = eq_thre.to(tl.int32).cumsum(0)

            concat_mask = tl.cat(over_thre, eq_thre, True)
            concat_input = tl.cat(input_idx, input_idx, True)
            concat_pointer_matrix = tl.cat(indices_base + sum + topk_idx - 1, s_input_ids_base + thre_bin_sum + thre_bin_idx - 1, True)

            # tl.debug_barrier()
            tl.store(concat_pointer_matrix, concat_input, mask = concat_mask)
                    
            thre_bin_sum += thre_bin_cur_sum
            sum += cur_sum

    if l_new_topk > 0:
        ss = tl.cdiv(l_new_topk, BK)
        for s in range(ss):
            s_input_idx = s * BK + tl.arange(0, BK)
            s_input_idx_mask = s_input_idx < l_new_topk
            input_idx = tl.load(s_input_ids_base + s_input_idx, s_input_idx_mask)
            tl.store(indices_base + sum + tl.arange(0, BK), input_idx, mask = s_input_idx_mask) # 这一句非常慢
            sum += BK


def bucket_sort_topk(inputs, starts, ends, topk):
    B, S = inputs.shape
    K = topk
    HISTOGRAM_SIZE = 256
    SMEM_INPUT_SIZE = 4096
    indices = torch.ones(B, topk, dtype=torch.int32, device=inputs.device) * -1
    # s_histogram = torch.zeros(B, HISTOGRAM_SIZE, dtype=torch.int32, device=inputs.device)
    s_input_idx = torch.zeros(B, SMEM_INPUT_SIZE, dtype=torch.int32, device=inputs.device)
    grid = (B,)
    kernel_bucket_sort_topk[grid]( # grid(B, BS)
        inputs, # (B, S) 注意，没有 H，因为MLA基于MQA和MHA而非GQA
        indices, # (B, K) topk 索引数组
        # s_histogram,
        s_input_idx,
        starts,
        ends,
        S, # sequence length
        K, # k of topk
        HISTOGRAM_SIZE,
        SMEM_INPUT_SIZE
    )
    return indices
