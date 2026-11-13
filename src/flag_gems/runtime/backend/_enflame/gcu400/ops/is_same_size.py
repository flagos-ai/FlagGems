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

logger = logging.getLogger(__name__)


def is_same_size(self: torch.Tensor, other: torch.Tensor) -> bool:
    """Implements ``aten::is_same_size(Tensor self, Tensor other) -> bool``.

    Returns whether the two tensors have identical shapes. ``is_same_size`` is
    a metadata-only operator: the answer is fully determined by the sizes of
    both operands, which the host already knows, and it never depends on
    tensor data. ATen's reference implementation likewise compares sizes on
    the host, so evaluating the comparison here returns the identical answer
    while skipping the staging of the shape vectors to the device, the
    reduction kernel launch, and the device-to-host sync of the flag --
    roughly 100us of round-trip overhead per call on GCU.
    """
    # gcu400 特化模块以短名 `gcu400.ops.*` 导入，accuracy_utils.gems_log_prefix
    # 据此推导出公共前缀 "GEMS"；日志文本必须与推导结果一致，否则工程用例失败。
    logger.debug("GEMS IS_SAME_SIZE")

    if not self.ndim:
        return not other.ndim

    # torch.Size is a tuple subclass, so this is a plain elementwise sizes
    # comparison -- exactly the check ATen performs.
    return self.shape == other.shape
