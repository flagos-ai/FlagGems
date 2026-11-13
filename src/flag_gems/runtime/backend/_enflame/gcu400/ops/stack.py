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

"""GCU400 专属 ``aten::stack`` 实现。

原生 ATen 的 ``stack`` 就是 ``_stack`` 的薄封装（组合分解），本模块沿用
同一分解：直接组合复用 GCU400 已优化的 ``_stack``（乘法寻址三路径拷贝，
legacy int64 除法交织 kernel 仅作受限 dtype 回退）。此前 ``stack`` 自带
一份与 ``_stack`` 旧版逐字节相同的 legacy kernel，在新工具链上 16.7M 元素
档位退化到 ~4ms（vs aten ~0.07ms），复用后与 ``_stack`` 同性能；同时继承
``empty_strided`` 分配（省掉 GEMS ``empty`` override 的 zero-init 额外
launch）、复数搬运、零体积早退与无条件的 dim 越界校验。
"""

import logging
from typing import List, Tuple, Union

import torch

from ._stack import _stack

logger = logging.getLogger(__name__)


def stack(
    tensors: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]], dim: int = 0
) -> torch.Tensor:
    logger.debug("GEMS_ENFLAME STACK")
    return _stack(tensors, dim)
