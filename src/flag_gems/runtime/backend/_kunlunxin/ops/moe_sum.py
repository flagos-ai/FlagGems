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
def _moe_sum_flat_copy_kernel(src_ptr, dst_ptr, BLOCK: tl.constexpr):
    """无掩码的 1D 扁平拷贝（输入输出均已 pad 到 BLOCK 倍数 / 或恰整除）。"""
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(dst_ptr + off, tl.load(src_ptr + off))


@libentry()
@triton.jit
def _moe_sum_direct_kernel(
    input_ptr,
    output_ptr,
    H: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_M: tl.constexpr,
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
    tl.store(
        output_ptr + m[:, None] * H + h[None, :], acc.to(input_ptr.dtype.element_ty)
    )


@libentry()
@triton.jit
def _moe_sum_tree_kernel(
    src_ptr,
    dst_ptr,
    H2: tl.constexpr,  # padded hidden width（BLOCK_H 倍数）
    SRC_S: tl.constexpr,
    DST_S: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_H: tl.constexpr,
    TILE_M: tl.constexpr,
    USE_1D: tl.constexpr,
    FINAL: tl.constexpr,
):
    """归约树一层：src (M, SRC_S, H2) -> dst (M, DST_S, H2)，FINAL 时 dst (M, H2)。

    XPU 上逐行 strided 读（256B/行交错）只有 ~7GB/s（实测），而 16 行整段的
    连续 1D load（BLOCK_E×H2 连续元素）+ reshape + tl.dot 列归约实测 169GB/s
    （HARNESS_SUMMARY 连续块 DMA 规律）。H2>1024（lanes 超上限）时退化为
    2D-tile（实测 ~81GB/s），避免跨行错位。tl.sum(axis=0) 在 XPU 编译崩溃，
    tl.dot 是绕过方案；默认 tf32 路径有 ~5e-4 误差，必须 input_precision="ieee"
    （实测 1e-5）。全无掩码：T（行数）与 hidden 由 wrapper 补齐。"""
    pid_m = tl.program_id(0)
    pid_es = tl.program_id(1)
    pid_h = tl.program_id(2)
    m0 = pid_m * TILE_M
    e0 = pid_es * BLOCK_E
    h0 = pid_h * BLOCK_H
    ones = tl.full((1, BLOCK_E), 1.0, dtype=tl.float32)
    for mm in tl.static_range(0, TILE_M):
        m = m0 + mm
        if USE_1D:
            off = (m * SRC_S + e0) * H2 + tl.arange(0, BLOCK_E * H2)
            v = tl.load(src_ptr + off).to(tl.float32)
            v2 = tl.reshape(v, (BLOCK_E, H2))
            acc = tl.dot(ones, v2, input_precision="ieee")
        else:
            e2 = e0 + tl.arange(0, BLOCK_E)
            h2 = h0 + tl.arange(0, BLOCK_H)
            off = (m * SRC_S + e2)[:, None] * H2 + h2[None, :]
            v = tl.load(src_ptr + off).to(tl.float32)
            acc = tl.dot(ones, v, input_precision="ieee")  # (1, BLOCK_H)
        if FINAL:
            tl.store(
                dst_ptr + m * H2 + h0 + tl.arange(0, H2 if USE_1D else BLOCK_H),
                tl.reshape(acc, (H2 if USE_1D else BLOCK_H,)).to(
                    src_ptr.dtype.element_ty
                ),
            )
        else:
            tl.store(
                dst_ptr
                + (m * DST_S + pid_es) * H2
                + h0
                + tl.arange(0, H2 if USE_1D else BLOCK_H),
                tl.reshape(acc, (H2 if USE_1D else BLOCK_H,)),
            )


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
            out = torch.empty(
                (M, H), dtype=output_work.dtype, device=output_work.device
            )
        # T==1 且数据量适中：moe_sum 退化为纯拷贝。小 H（如 H==1）时逐行
        # kernel 每程序 lanes 太少（16-lane 程序实测 ~9us/程序，625 程序 ~5.5ms），
        # 扁平大块 1D 拷贝（16384 lanes/程序）能把 launch 开销压缩两个数量级。
        # 大 N（超过 16M 元素）仍走 direct kernel（避免 2x padding 分配）；
        # 极小 N 且 H>1 时 2 次 alloc 的 pad 路径反而不如 direct 快。
        # 注意源必须用未 pad 的 input_work（H 补齐后的 work 行宽与 M*H 错位）。
        n_flat = M * hidden_size
        if (
            topk == 1
            and n_flat <= (1 << 24)
            and (hidden_size == 1 or n_flat >= (1 << 16))
        ):
            block = 16384
            n_pad = (n_flat + block - 1) // block * block
            if n_pad == n_flat:
                _moe_sum_flat_copy_kernel[(n_flat // block,)](
                    input_work.view(-1), out.view(-1), BLOCK=block
                )
            else:
                src_w = torch.empty(
                    (n_pad,), dtype=input_work.dtype, device=input_work.device
                )
                torch.ops.aten._copy_from(input_work.view(-1), src_w[:n_flat], False)
                dst_w = torch.empty(
                    (n_pad,), dtype=output_work.dtype, device=output_work.device
                )
                _moe_sum_flat_copy_kernel[(n_pad // block,)](src_w, dst_w, BLOCK=block)
                torch.ops.aten._copy_from(dst_w[:n_flat], out.view(-1), False)
        elif topk <= STATIC_UNROLL:
            block_h = H if H <= 4096 else 4096
            tile_m = _tile_m_for(M, block_h)
            _moe_sum_direct_kernel[(M // tile_m, H // block_h)](
                work,
                out,
                H=H,
                TOPK=topk,
                BLOCK_H=block_h,
                TILE_M=tile_m,
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
                    torch.zeros(
                        (M, dst_s, H), dtype=torch.float32, device=input_work.device
                    )
                )
            # 树层连续大块读：H2<=1024 走 1D+reshape（169GB/s），更大走 2D-tile
            block_e = STATIC_UNROLL
            use_1d = H <= 1024
            block_h = H if use_1d else 1024
            tile_m = _tile_m_for(M, block_e * block_h)
            for li in range(n_layers):
                s_cur, dst_s = layers[li]
                src_cur = work if li == 0 else buffers[li - 1]
                is_final = li == n_layers - 1
                dst_cur = out if is_final else buffers[li]
                grid = (
                    M // tile_m,
                    (s_cur + block_e - 1) // block_e,
                    1 if use_1d else H // block_h,
                )
                _moe_sum_tree_kernel[grid](
                    src_cur,
                    dst_cur,
                    H2=H,
                    SRC_S=s_cur,
                    DST_S=dst_s,
                    BLOCK_E=block_e,
                    BLOCK_H=block_h,
                    TILE_M=tile_m,
                    USE_1D=use_1d,
                    FINAL=is_final,
                )
        if out is not output_work:
            torch.ops.aten._copy_from(out[:, :hidden_size], output_work, False)
    if output_work is not output:
        output.copy_(output_work)
