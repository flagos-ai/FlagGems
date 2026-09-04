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
import logging

import torch

logger = logging.getLogger("flag_gems." + __name__)


# XPU (xpytorch) 上 aten._nested_view_from_buffer / _copy 的定制实现会断言
# buffer_storage_size == 组件元素总数，且由 _nested_view_from_buffer 构造出的
# 嵌套张量后续读取（unbind / index）会直接段错误；因此 Kunlunxin 后端唯一可用
# 的嵌套张量构造方式是 torch.nested 家族 API。
#
# 性能修复（相对默认 _metax 实现：逐组件 as_strided + clone + nested_tensor）：
#   1. torch.nested.nested_tensor 在 use_gems 场景下内部多次触发 FlagGems 的
#      cat/empty/copy override，稳态延迟约 0.7ms 且首个 shape 编译超 7ms+；
#   2. 改用 empty_strided + aten._copy_from 完成一次整块拷贝快照（这两个原语
#      FlagGems 均不 override，直达 vendor 拷贝引擎、无 Triton 首编译污染），
#      再以 torch.nested.as_nested_tensor 视图组装嵌套张量（view 路径不经过
#      被 override 的 cat），保持 _copy 的拷贝语义：use_gems 稳态约 0.4ms，
#      无逐组件拷贝 launch。
def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_BUFFER_COPY")

    snapshot = torch.empty_strided(
        self.shape, self.stride(), dtype=self.dtype, device=self.device
    )
    torch.ops.aten._copy_from(self, snapshot, False)

    num_components = nested_size.shape[0]
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
