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

# Capture the native kernels before FlagGems installs its CUDA registrations.
# The explicit keyset reaches Kunlunxin's implementation without recursion.
_NATIVE_LOGADDEXP = torch.library.get_kernel("aten::logaddexp", "CUDA")
_NATIVE_LOGADDEXP_OUT = torch.library.get_kernel("aten::logaddexp.out", "CUDA")
_CUDA_KEYSET = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)


def logaddexp(self, other):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP_NATIVE")
    return _NATIVE_LOGADDEXP.call_boxed(_CUDA_KEYSET, self, other)


def logaddexp_out(self, other, out):
    logger.debug("GEMS_KUNLUNXIN LOGADDEXP_OUT_NATIVE")
    return _NATIVE_LOGADDEXP_OUT.call_boxed(_CUDA_KEYSET, self, other, out=out)
