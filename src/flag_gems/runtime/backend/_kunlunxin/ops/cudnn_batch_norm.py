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


def cudnn_batch_norm(
    input,
    weight,
    bias,
    running_mean=None,
    running_var=None,
    training=True,
    exponential_average_factor=0.1,
    epsilon=1e-5,
):
    """aten::cudnn_batch_norm, provided for the vendor torch_xmlir XPU build.

    NOTE (kunlunxin / XPU, 2026-09-06): why this file exists at all.

    The vendor torch (2.9.0+cu129 + torch_xmlir SYMBOL_REWRITE XPU shim) ships
    NO kernel for the cuDNN-family ATen ops: `aten::cudnn_batch_norm` is only
    reachable through the ts_eager_fallback, which raises
        NotImplementedError: Could not run 'aten::cudnn_batch_norm' with
        arguments from the 'CPU' backend.
    (`torch._C._dispatch_has_kernel_for_dispatch_key` is False for CUDA / XPU /
    PrivateUse1 / CPU alike; `test_cudnn_convolution.py` fails identically.)
    `tests/test_cudnn_batch_norm_backward.py` calls this op OUTSIDE any
    flag_gems.use_gems() context just to produce save_mean / save_var, so it can
    never reach the (already correct) vendor `cudnn_batch_norm_backward` kernel
    without a real forward implementation.

    This file provides one: the training forward pass is computed with the
    vendor's own `aten::native_batch_norm` XPU kernel (same statistics path the
    rest of the platform uses), and its `save_invstd` output is converted back
    to the `save_var` semantics of `cudnn_batch_norm`:
        save_invstd == 1 / sqrt(var + eps)  =>  save_var = 1 / save_invstd^2 - eps
    so `(output, save_mean, save_var, reserve)` matches cuDNN's training-mode
    statistics (both are the biased batch variance / batch mean).  Everything
    runs on the XPU device; there is no CPU fallback.

    Registration is a plain `torch.library.impl` at module import time so the op
    is active outside flag_gems.use_gems() (the accuracy and benchmark tests
    call it before opening the GEMS context).  The guard makes the registration
    idempotent and future-proof: if a later vendor torch (or another override)
    ever provides the op, this shim silently steps aside.
    """
    logger.debug("GEMS_KUNLUNXIN CUDNN_BATCH_NORM (forward shim)")
    if input is None or weight is None or bias is None:
        raise ValueError("cudnn_batch_norm requires input, weight and bias tensors")
    out, save_mean, save_invstd = torch.ops.aten.native_batch_norm(
        input,
        weight,
        bias,
        running_mean,
        running_var,
        bool(training),
        float(exponential_average_factor),
        float(epsilon),
    )
    save_var = (1.0 / (save_invstd * save_invstd) - float(epsilon)).to(
        save_invstd.dtype
    )
    reserve = torch.empty(0, device=input.device, dtype=input.dtype)
    return out, save_mean, save_var, reserve


if not torch._C._dispatch_has_kernel_for_dispatch_key(
    "aten::cudnn_batch_norm", torch._C.DispatchKey.CUDA
):
    torch.library.impl("aten::cudnn_batch_norm", "CUDA", cudnn_batch_norm)