import logging

import torch

from ..utils.tle_copy import tle_copy

logger = logging.getLogger("flag_gems." + __name__)


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
#      与参考语义一致。整个快速路径只使用元数据原语（empty_strided /
#      _nested_view_from_jagged）+ gem 自己的 copy_（Triton），零主机同步，
#      use_gems 与裸调用耗时相同（~0.22ms）。⚠️ 2026-09-14：原先用
#      `aten::_copy_from`（vendor 拷贝引擎）做快照，属被禁的 vendor 委托，
#      已改为 `copy_()`（走 FlagGems copy override）。
#   3. 限制：jagged 组件为连续（stride-1）1-D 视图，故仅当 self 为 1-D、
#      nested_size 为 (N,1) int64、strides 全 1、offsets 为 int64 时走快速路径；
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
        and all(s == 1 for s in nested_strides.reshape(-1).tolist())
    ):
        # One flat copy of the whole buffer (copy semantics of the op; the
        # nested tensor then is a vi ew of `values`).
        values = torch.empty_strided(
            self.shape, self.stride(), dtype=self.dtype, device=self.device
        )
        values.copy_(self)
        # Jagged offsets must have num_components+1 entries; with explicit
        # `lengths` the trailing entry is not used for component sizes, so the
        # input offsets (padded by one element) are passed through unchanged.
        full_offsets = torch.empty_strided(
            (num_components + 1,), (1,), dtype=torch.int64, device=self.device
        )
        full_offsets[:num_components].copy_(offsets)
        full_offsets[num_components:].copy_(offsets[:1])
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
