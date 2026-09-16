import logging

import torch
import triton
import triton.language as tl

from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems." + __name__)

# 载荷拷贝的分块（元素数）。B=8192 是在 0.4–0.8MB 载荷上验证过的定值；未扫参。
_CPO_BLOCK = 8192

# 元数据补齐走标量循环 kernel 的组件数上限。
# 实测（2026-09-15，同窗口 A/B，三轮复判）：NC≤16 比 `copy_` 快 ~23%，NC≥24 起反而慢
# （标量循环 ~1.4µs/次迭代，是串行延迟不是带宽；24→1.03× / 28→1.17× / 32→1.24×）
# ⇒ 取 **16** 作保守阈值；超过则批量走改动前的 `copy_`（零回退）。
_PAD_SCALAR_MAX = 16


@triton.jit
def _copy_payload_kernel(self_ptr, values_ptr, NP, S0, BLOCK: tl.constexpr):
    """载荷拷贝：按 `self` 的 stride 拷 NP 个元素。

    `values` 是 `empty_strided(self.shape, self.stride())`，两者布局一致 ⇒ 同一套索引对两边都成立。
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NP
    off = idx * S0
    tl.store(values_ptr + off, tl.load(self_ptr + off, mask=m), mask=m)


@triton.jit
def _pad_offsets_kernel(off_ptr, fo_ptr, NC, SO):
    """把 `offsets` 写进 `full_offsets[:NC]`，并补末位 `full_offsets[NC] = offsets[0]`。

    ⚠️ 用**标量循环**（无 constexpr、无 mask）是有意的：
    2026-09-15 的正确性判定发现，把「小载荷 mask」与「masked 向量 store」写进**同一个 kernel** 时，
    元数据的中段会被静默漏写（`numel ≤ 256 & NC ≥ 2` 必现；ablation 显示元数据换标量循环即好，
    机制未读到 IR、暂按"拆开写"规避）。两件事拆成两次启动后，本 kernel 的形状与 ablation 里
    "好"的那一版一致。标量循环同时避免了 `META` 作 constexpr 导致的 per-组件数重编译。
    """
    for i in range(0, NC):
        tl.store(fo_ptr + i, tl.load(off_ptr + i * SO))
    tl.store(fo_ptr + NC, tl.load(off_ptr))


# XPU (xpytorch) 上 aten._nested_view_from_buffer / _copy 的定制实现会断言
# buffer_storage_size == 组件元素总数，且由 _nested_view_from_buffer 构造出的
# 嵌套张量后续读取（unbind / index）会直接段错误；因此 Kunlunxin 后端唯一可用
# 的嵌套张量构造方式是 torch.nested 家族 API。
#
# 性能修复（相对上一版：empty_strided + _copy_from 快照 + as_nested_tensor 组装）：
#   1. 上一版的 `torch.nested.as_nested_tensor` 内部走 `_nested_tensor_from_tensor_list`
#      → `torch.cat`，而 `cat` 恰是被 FlagGems override 的算子：在 use_gems 下
#      3 个不等长组件命中 cat.py 的通用 dim-0 路径（3 次 Triton copy launch），
#      仅此一项即 ~0.2ms；加上 9 次 `.item()` 主机同步（~0.13ms），use_gems
#      稳态 ~0.4ms；
#   2. 改用 **jagged layout** 的 `_nested_view_from_values_offsets_lengths` 视图
#      构造（`torch._nested_view_from_jagged`）：组件长度（lengths）显式传入，
#      因此任意 offsets（含空洞/重叠）都直接映射到 `values[offsets[i]:+len_i]`，
#      与参考语义一致。整套路径只使用元数据原语（empty_strided /
#      _nested_view_from_jagged）+ 我们自己写的 Triton 搬运。
#      ⚠️ 2026-09-15：搬运原先拆成 3 次 `copy_()`（载荷 + 两次 32B 元数据），
#      在 use_gems 下每次要多付 ~24µs 的 Python 派发层（3 次 copy_ 的派发合计 **≈107µs**，隔离实测；
#      整个「3→1 融合」探针实测总省 ~150µs；而**设备侧总共只有 ~6µs** —— 90 个 kernel / 30 次调用）
#      ⇒ 改为**两次裸 kernel 启动**（载荷 / 元数据各一次；不合成一次的理由见 `_pad_offsets_kernel`）。
#   3. 限制：jagged 组件为连续（stride-1）1-D 视图，故仅当 self 为 1-D、
#      nested_size 为 (N,1) int64、strides 全 1、offsets 为 1-D int64 且长度 ≥ 组件数，
#      以及 `numel * stride` 不越 int32 索引范围时走快速路径；
#      其他情况回退到通用 `as_nested_tensor` 路径（保留任意 stride/维度语义）。
def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_BUFFER_COPY")
    num_components = nested_size.shape[0]

    if (
        self.dim() == 1
        and nested_size.dim() == 2
        and nested_size.shape[1] == 1
        and nested_size.dtype == torch.int64
        and nested_strides.dtype == torch.int64
        and offsets.dtype == torch.int64
        and offsets.dim() == 1
        and offsets.numel() >= max(1, num_components)
        and all(s == 1 for s in nested_strides.reshape(-1).tolist())
        and self.numel() * max(1, self.stride(0)) < 2**31
    ):
        # 载荷一次拷完（op 的 copy 语义；嵌套张量随后是 `values` 的一个视图），
        # 元数据（offsets 补齐到 num_components+1 位）由第二次启动完成。
        values = torch.empty_strided(
            self.shape, self.stride(), dtype=self.dtype, device=self.device
        )
        full_offsets = torch.empty_strided(
            (num_components + 1,), (1,), dtype=torch.int64, device=self.device
        )
        _copy_payload_kernel[(max(1, triton.cdiv(self.numel(), _CPO_BLOCK)),)](
            self, values, self.numel(), self.stride(0), _CPO_BLOCK
        )
        if num_components <= _PAD_SCALAR_MAX:
            _pad_offsets_kernel[(1,)](
                offsets, full_offsets, num_components, offsets.stride(0)
            )
        else:
            # 大组件数：批量走改动前的 `copy_`（标量循环在大 NC 上慢），**末位仍用上面的 kernel**。
            # 不用 `full_offsets[n:].copy_(offsets[:1])`：1 元素张量的 `is_contiguous()` 恒为真，
            # 会让 gem copy_ 的 tle 快路径误判（`TensorDescriptor` 断言最后一维 stride==1），
            # **冷跑必抛** —— 这是 copy_ 侧的既有缺陷，本 kernel 按 `stride(0)` 寻址绕开它。
            full_offsets[:num_components].copy_(offsets)
            _pad_offsets_kernel[(1,)](
                offsets, full_offsets[num_components:], 0, offsets.stride(0)
            )
        from torch.nested._internal.nested_tensor import (
            nested_view_from_values_offsets_lengths,
        )

        return nested_view_from_values_offsets_lengths(
            values,
            full_offsets,
            nested_size[:, 0],
            ragged_idx=1,
            min_seqlen=None,
            max_seqlen=None,
        )

    # Generic fallback: per-component as_strided views of a snapshot copy.
    snapshot = torch.empty_strided(
        self.shape, self.stride(), dtype=self.dtype, device=self.device
    )
    snapshot.copy_(self)

    components = []
    for i in range(num_components):
        size_i = int(nested_size[i].item())
        stride_i = (
            int(nested_strides[i].item())
            if nested_strides.ndim > 1
            else int(nested_strides[i].item())
        )
        offset_i = int(offsets[i].item())
        components.append(snapshot.as_strided((size_i,), (stride_i,), offset_i))

    return torch.nested.as_nested_tensor(components)


__all__ = ["_nested_view_from_buffer_copy"]
