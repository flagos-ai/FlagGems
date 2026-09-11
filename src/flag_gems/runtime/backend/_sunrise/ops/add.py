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
import triton

from flag_gems.utils import pointwise_dynamic
from flag_gems.utils.codegen_config_utils import CodeGenConfig
from flag_gems.utils.pointwise_dynamic import ComplexMode

logger = logging.getLogger(__name__)


config_for_general = CodeGenConfig(
    1024,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=False,
    # num_warps=2
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_for_general,
)
@triton.jit
def add_func(x, y, alpha):
    return x + y * alpha


config_for_broadcast = CodeGenConfig(
    128,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    # num_warps=4
)


@pointwise_dynamic(
    is_tensor=[True, True, False],
    promotion_methods=[(0, 1, "DEFAULT")],
    config=config_for_broadcast,
)
@triton.jit
def add_func_broadcast(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[True, False, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_tensor_scalar(x, y, alpha):
    return x + y * alpha


@pointwise_dynamic(
    is_tensor=[False, True, False], promotion_methods=[(0, 1, "DEFAULT")]
)
@triton.jit
def add_func_scalar_tensor(x, y, alpha):
    return x + y * alpha


def get_best_strided_output_tensor(A, B):
    def get_best_strides(A, B, broadcast_shape):
        if A.shape == broadcast_shape:
            return A.stride()
        elif B.shape == broadcast_shape:
            return B.stride()
        return None

    broadcast_shape = torch.broadcast_shapes(A.shape, B.shape)
    dtype = torch.float32
    out = torch.empty(broadcast_shape, device=A.device, dtype=dtype)
    best_stride = get_best_strides(A, B, broadcast_shape)
    if best_stride is not None:
        out = out.as_strided(broadcast_shape, best_stride)
    return out


def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0


def should_use_broadcast_configs(A, B):
    # In scenarios where broadcasting is involved and the last two dimensions
    # of the two input tensors are the same, we use 1D tiling with a smaller
    # max_tile_size config for better performance.
    need_broadcast = A.shape != B.shape
    has_equal_last_dimentions = (
        len(A.shape) >= 2 and len(B.shape) >= 2 and A.shape[-2:] == B.shape[-2:]
    )
    return (
        need_broadcast
        and has_equal_last_dimentions
        and not is_power_of_two(A.shape[-1])
        and torch.result_type(A, B) in [torch.float16, torch.float32]
    )


# Register complex support (elementwise)
add_func.register_complex(mode=ComplexMode.ELEMENTWISE)
add_func_tensor_scalar.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=add_func
)
add_func_scalar_tensor.register_complex(
    mode=ComplexMode.ELEMENTWISE, tensorize_scalars=True, fallback_target=add_func
)


def _view_as_real_ptpu_safe(x: torch.Tensor) -> torch.Tensor:
    """`torch.view_as_real(x)` with a CPU bounce when x is on PTPU.

    [sunrise fix] PTPU lacks `aten::view_as_real`. The surrounding complex
    branch uses the result only as a read-only input to the triton `add_func`
    kernel (which IS PTPU-native), and the subsequent `.to(common_dtype)` would
    materialize a non-aliasing copy anyway — so it is safe to break alias
    semantics here. Per the FlagGems Sunrise skill, do not generically
    monkey-patch view_as_real (alias/view primitive). Compute stays on PTPU.
    """
    try:
        return torch.view_as_real(x)
    except NotImplementedError:
        if x.device.type != "ptpu":
            raise
        return torch.view_as_real(x.cpu()).to(x.device)


def _view_as_complex_ptpu_safe(x: torch.Tensor) -> torch.Tensor:
    """`torch.view_as_complex(x)` with a CPU bounce when x is on PTPU.

    See `_view_as_real_ptpu_safe` above. Used here to recompose the complex
    output after the PTPU-native real-domain `add_func(Ar, Br, alpha)` finishes.
    """
    try:
        return torch.view_as_complex(x)
    except NotImplementedError:
        if x.device.type != "ptpu":
            raise
        return torch.view_as_complex(x.cpu()).to(x.device)


def _scalar_complex_as_real_ptpu_safe(
    scalar, complex_dtype: torch.dtype, target_shape, device: torch.device
) -> torch.Tensor:
    """Broadcast a python scalar to `view_as_real`-shaped tensor on `device`.

    [sunrise fix] The natural code path is

        torch.view_as_real(
            torch.tensor(scalar, dtype=complex_dtype, device=device).expand_as(ref)
        )

    On PTPU this dies at the `view_as_real` step (no kernel) and the obvious
    CPU fallback (`.cpu()`) also dies because PTPU's `direct_copy_kernel_ptpu`
    has no entry for `ComplexHalf` / `ComplexFloat`. So instead we build the
    complex scalar AND take its real view ENTIRELY on CPU, then only move the
    final real-dtype tensor onto PTPU (which the device's copy_ DOES support).
    """
    cpu_scalar = torch.tensor(scalar, dtype=complex_dtype, device="cpu").expand(
        target_shape
    )
    cpu_real = torch.view_as_real(cpu_scalar).contiguous()
    if device.type == "cpu":
        return cpu_real
    return cpu_real.to(device)


def add(A, B, *, alpha=1):
    logger.debug("GEMS_SUNRISE ADD")
    A_is_complex = (isinstance(A, torch.Tensor) and A.is_complex()) or isinstance(
        A, complex
    )
    B_is_complex = (isinstance(B, torch.Tensor) and B.is_complex()) or isinstance(
        B, complex
    )
    if A_is_complex or B_is_complex:
        if A_is_complex and B_is_complex:
            Ar = _view_as_real_ptpu_safe(A)
            Br = _view_as_real_ptpu_safe(B)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return _view_as_complex_ptpu_safe(out_real).to(torch.result_type(A, B))
        elif A_is_complex and not B_is_complex:
            Ar = _view_as_real_ptpu_safe(A)
            if isinstance(B, torch.Tensor):
                Br = _view_as_real_ptpu_safe(B.to(A.dtype))
            else:
                Br = _scalar_complex_as_real_ptpu_safe(B, A.dtype, A.shape, A.device)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return _view_as_complex_ptpu_safe(out_real).to(torch.result_type(A, B))
        else:
            Br = _view_as_real_ptpu_safe(B)
            if isinstance(A, torch.Tensor):
                Ar = _view_as_real_ptpu_safe(A.to(B.dtype))
            else:
                Ar = _scalar_complex_as_real_ptpu_safe(A, B.dtype, B.shape, B.device)
            common_dtype = torch.promote_types(Ar.dtype, Br.dtype)
            Ar, Br = Ar.to(common_dtype), Br.to(common_dtype)
            out_real = add_func(Ar, Br, alpha)
            return _view_as_complex_ptpu_safe(out_real).to(torch.result_type(A, B))
    elif isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        if should_use_broadcast_configs(A, B):
            out = get_best_strided_output_tensor(A, B)
            add_func_broadcast(A, B, alpha, out0=out)
            return out.to(torch.result_type(A, B))
        else:
            return add_func(A, B, alpha)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha)
    elif isinstance(B, torch.Tensor):
        return add_func_scalar_tensor(A, B, alpha)
    else:
        return torch.tensor(A + B * alpha)


def add_(A, B, *, alpha=1):
    logger.debug("GEMS_SUNRISE ADD_")
    if isinstance(A, torch.Tensor) and isinstance(B, torch.Tensor):
        if B.device != A.device:
            B = B.to(A.device)
        return add_func(A, B, alpha, out0=A)
    elif isinstance(A, torch.Tensor):
        return add_func_tensor_scalar(A, B, alpha, out0=A)
    # elif isinstance(B, torch.Tensor):
    #     return add_func_scalar_tensor(A, B, alpha, out0=A)
    else:
        raise ValueError("Unreachable.")
