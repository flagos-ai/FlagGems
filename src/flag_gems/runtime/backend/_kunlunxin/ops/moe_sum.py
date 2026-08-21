import logging

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

logger = logging.getLogger(__name__)

# 每程序静态融合的行数上限：超过后走多级静态归约树（避免动态循环在 XPU 上的
# 巨大惩罚，也避免 static_range 大展开的 IR 爆炸 / uni_sram 编译失败）。
STATIC_UNROLL = 16

# XPU 上 masked load/store（尤其尾部块）不可靠：mask 可能被忽略且越界读，
# 触发 KL_XID_KERNEL_EXCEPTION（HARNESS_SUMMARY §2.5）。因此 kernel 一律
# 全无掩码，由 wrapper 把输入 padding 到 16 的倍数行 / pow2 hidden（尾部全零）。


@libentry()
@triton.jit
def _moe_sum_direct_kernel(
    input_ptr, output_ptr,
    H: tl.constexpr, TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr, TILE_M: tl.constexpr,
):
    """TOPK <= STATIC_UNROLL 时直接融合归约：每程序 (TILE_M, TOPK, BLOCK_H)。
    全无掩码（输入已由 wrapper 保证整除/补齐）。"""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    m = pid_m * TILE_M + tl.arange(0, TILE_M)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    base = m[:, None] * (TOPK * H) + h[None, :]
    acc = tl.zeros((TILE_M, BLOCK_H), dtype=tl.float32)
    for e in tl.static_range(0, TOPK):
        acc += tl.load(input_ptr + base + e * H).to(tl.float32)
    tl.store(output_ptr + m[:, None] * H + h[None, :], acc.to(input_ptr.dtype.element_ty))


@libentry()
@triton.jit
def _moe_sum_tree_kernel(
    src_ptr, dst_ptr,
    H: tl.constexpr, SRC_S: tl.constexpr, DST_S: tl.constexpr,
    BLOCK_H: tl.constexpr, TILE_M: tl.constexpr, FINAL: tl.constexpr,
):
    """归约树一层：src (M, SRC_S, H) -> dst (M, DST_S, H)，FINAL 时 dst (M, H)。
    每程序静态融合 16 行，行块划分放 grid 第三维；全无掩码（补齐由 wrapper 保证）。"""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_es = tl.program_id(2)
    m = pid_m * TILE_M + tl.arange(0, TILE_M)
    h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    e0 = pid_es * 16
    acc = tl.zeros((TILE_M, BLOCK_H), dtype=tl.float32)
    for i in tl.static_range(0, 16):
        acc += tl.load(src_ptr + (m[:, None] * SRC_S + (e0 + i)) * H + h[None, :]).to(tl.float32)
    if FINAL:
        tl.store(dst_ptr + m[:, None] * H + h[None, :], acc.to(src_ptr.dtype.element_ty))
    else:
        tl.store(dst_ptr + (m[:, None] * DST_S + pid_es) * H + h[None, :], acc)


def _next_pow2(x):
    p = 1
    while p < x:
        p *= 2
    return p


def _tile_m_for(M, block_h):
    """整除 M 且 (tile_m*block_h) <= 16384 的最大 2 的幂，保证 token 维无掩码。"""
    for tm in (64, 32, 16, 8, 4, 2, 1):
        if M % tm == 0 and tm * block_h <= 16384:
            return tm
    return 1


def _prep_input(x):
    """补齐行数到 16 的倍数（topk>16 时）与 hidden 到 pow2（新增区全零），
    kernel 全程无掩码。"""
    M, T, H = x.shape
    T2 = T
    if T > STATIC_UNROLL:
        T2 = ((T + STATIC_UNROLL - 1) // STATIC_UNROLL) * STATIC_UNROLL
    H2 = _next_pow2(H)
    if T2 == T and H2 == H:
        return x
    y = torch.zeros((M, T2, H2), dtype=x.dtype, device=x.device)
    torch.ops.aten._copy_from(x, y[:, :T, :H], False)
    return y


def moe_sum(input, output):
    logger.debug("GEMS_KUNLUNXIN MOE_SUM")
    input_work = input.contiguous()
    output_work = output if output.is_contiguous() else output.new_empty(output.shape)
    M, topk, hidden_size = input_work.shape
    if M == 0 or topk == 0 or hidden_size == 0:
        if output_work is not output:
            output.copy_(output_work)
        return
    with torch_device_fn.device(input.device):
        work = _prep_input(input_work)
        H = work.shape[2]
        out = output_work
        if H != hidden_size:
            # hidden 补齐后 kernel 按 H 写输出，需在补齐宽度的输出缓冲上运行
            out = torch.empty((M, H), dtype=output_work.dtype, device=output_work.device)
        if topk <= STATIC_UNROLL:
            block_h = H if H <= 4096 else 4096
            tile_m = _tile_m_for(M, block_h)
            _moe_sum_direct_kernel[(M // tile_m, H // block_h)](
                work, out, H=H, TOPK=topk, BLOCK_H=block_h, TILE_M=tile_m,
            )
        else:
            s_src = work.shape[1]
            layers = []
            s = s_src
            while s > STATIC_UNROLL:
                nxt = (s + STATIC_UNROLL - 1) // STATIC_UNROLL
                nxt_pad = ((nxt + STATIC_UNROLL - 1) // STATIC_UNROLL) * STATIC_UNROLL
                layers.append((s, nxt_pad))
                s = nxt_pad
            layers.append((s, 0))
            n_layers = len(layers)
            buffers = []
            for li in range(n_layers - 1):
                _, dst_s = layers[li]
                buffers.append(
                    torch.zeros((M, dst_s, H), dtype=torch.float32, device=input_work.device)
                )
            block_h = H if H <= 4096 else 4096
            for li in range(n_layers):
                s_cur, dst_s = layers[li]
                src_cur = work if li == 0 else buffers[li - 1]
                is_final = (li == n_layers - 1)
                dst_cur = out if is_final else buffers[li]
                tile_m = _tile_m_for(M, block_h)
                grid = (
                    M // tile_m,
                    H // block_h,
                    (s_cur + STATIC_UNROLL - 1) // STATIC_UNROLL,
                )
                _moe_sum_tree_kernel[grid](
                    src_cur, dst_cur,
                    H=H, SRC_S=s_cur, DST_S=dst_s,
                    BLOCK_H=block_h, TILE_M=tile_m, FINAL=is_final,
                )
        if out is not output_work:
            torch.ops.aten._copy_from(out[:, :hidden_size], output_work, False)
    if output_work is not output:
        output.copy_(output_work)