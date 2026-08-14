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

import warnings

import torch


def _bitwise_and_scalar_tensor_fallback(A, B):
    # The native Kunlunxin composite materializes A on CPU before applying the
    # operation. AND is commutative, so reuse the native Tensor_Scalar overload.
    return torch.ops.aten.bitwise_and.Scalar(B, A)


def _bitwise_or_scalar_tensor_fallback(A, B):
    # The native Kunlunxin composite materializes A on CPU before applying the
    # operation. OR is commutative, so reuse the native Tensor_Scalar overload.
    return torch.ops.aten.bitwise_or.Scalar(B, A)


_scalar_tensor_fallback_lib = None


def ensure_scalar_tensor_fallbacks():
    global _scalar_tensor_fallback_lib
    if _scalar_tensor_fallback_lib is not None:
        return

    # Register one shared composite fallback for both operators. A CUDA-specific
    # FlagGems registration has higher dispatch priority and remains responsible
    # for calls made inside use_gems(). Keep this handle alive at module scope;
    # torch.library removes registrations when their Library is collected.
    lib = torch.library.Library("aten", "IMPL", "CompositeExplicitAutograd")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Warning only once for all operators,.*",
            category=UserWarning,
            module=r"torch\.library",
        )
        lib.impl("bitwise_and.Scalar_Tensor", _bitwise_and_scalar_tensor_fallback)
        lib.impl("bitwise_or.Scalar_Tensor", _bitwise_or_scalar_tensor_fallback)
    _scalar_tensor_fallback_lib = lib
