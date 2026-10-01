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
import triton.language as tl

from flag_gems import runtime
from flag_gems.utils import tl_extra_shim
from flag_gems.utils.shape_utils import MemOverlap, has_internal_overlapping
from flag_gems.utils.type_utils import ELEMENTWISE_TYPE_PROMOTION_KIND, type_promotion

logger = logging.getLogger("flag_gems.ops.special_zeta")

_pow = tl_extra_shim.pow
_SUPPORTED_COMPUTE_DTYPES = (torch.float32, torch.float64)
_BLOCK = 1024
_MAX_GRID = 65535
_PAD_VALUE = 2.0


@triton.jit
def _zeta_compute(x, q):
    """Cephes Euler--Maclaurin approximation (XPU-compilable rewrite).

    The generic kernel uses a `tl.sum` ballot ``while`` loop and per-term
    ``em_done`` early-termination ``tl.where``/``tl.abs`` guards; both make the
    TritonXPU backend crash (uni_sram OutOfResources) or hang during compile.
    Here the ballot loop is dropped (for q>0 the 9-step recurrence already
    leaves a=q+9>9 so the extra loop runs zero iterations; q<=0 results are
    overridden by the domain checks below) and the twelve Bernoulli tail terms
    are accumulated unconditionally (skipped terms are already negligible).
    """
    machep = 1.11022302462515654042e-16
    total = _pow(q, -x)
    a = q
    b = total
    direct_done = x < x
    for _ in tl.static_range(9):
        active = ~direct_done
        next_a = a + 1.0
        next_b = _pow(next_a, -x)
        next_total = total + next_b
        converged = (-machep * next_total < next_b) & (next_b < machep * next_total)
        a = tl.where(active, next_a, a)
        b = tl.where(active, next_b, b)
        total = tl.where(active, next_total, total)
        direct_done = direct_done | (active & converged)
    direct_result = total

    w = a
    total = total + b * w / (x - 1.0) - 0.5 * b
    product = 1.0 + 0.0 * x
    k = 0
    product = product * (x + k)
    b = b / w
    total = total + product * b / 12.0
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -720.0
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / 30240.0
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -1209600.0
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / 47900160.0
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -1.8924375803183791606e9
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / 7.47242496e10
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -2.950130727918164224e12
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / 1.1646782814350067249e14
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -4.5979787224074726105e15
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / 1.8152105401943546773e17
    k += 1
    product = product * (x + k)
    b = b / w
    k += 1
    product = product * (x + k)
    b = b / w
    total = total + product * b / -7.1661652561756670113e18
    result = tl.where(direct_done, direct_result, total)

    q_nonpositive = q <= 0.0
    q_integer = q == tl.floor(q)
    x_integer = x == tl.floor(x)
    result = tl.where(q_nonpositive & (~q_integer) & (~x_integer), float("nan"), result)
    result = tl.where(q_nonpositive & q_integer, float("inf"), result)
    result = tl.where(x < 1.0, float("nan"), result)
    result = tl.where((x == float("inf")) & (q == 1.0), 1.0, result)
    return tl.where(x == 1.0, float("inf"), result)


@triton.jit
def _special_zeta_flat_kernel(x, q, out, n_tiles, BLOCK: tl.constexpr):
    n_prog = tl.num_programs(0)
    tile = tl.program_id(0)
    while tile < n_tiles:
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        xv = tl.load(x + offsets)
        qv = tl.load(q + offsets)
        r = _zeta_compute(xv.to(tl.float32), qv.to(tl.float32))
        tl.store(out + offsets, r)
        tile += n_prog


def _promoted_dtype(x, q):
    _, result_dtype = type_promotion(
        x,
        q,
        type_promotion=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    )
    if result_dtype not in _SUPPORTED_COMPUTE_DTYPES:
        raise RuntimeError(
            "special_zeta kernel only supports a promoted float32 or float64 "
            f"dtype, but got {result_dtype}"
        )
    if result_dtype == torch.float64 and not runtime.device.support_fp64:
        raise RuntimeError(
            f"special_zeta does not support float64 on {runtime.device.vendor_name}"
        )
    return result_dtype


def _validate_tensor_devices(*tensors):
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise RuntimeError("special_zeta expected all tensors to be on the same device")
    return device


def _tensors_overlap(left, right):
    try:
        return torch._C._overlaps(left, right)
    except AttributeError:
        return left is right


def _is_exact_alias(left, right):
    if left is right:
        return True
    if left.device != right.device or left.dtype != right.dtype:
        return False
    return (
        left.untyped_storage().data_ptr() == right.untyped_storage().data_ptr()
        and left.storage_offset() == right.storage_offset()
        and left.shape == right.shape
        and left.stride() == right.stride()
    )


def _prepare_out(out, shape, device, inputs):
    if out.device != device:
        raise RuntimeError(
            f"special_zeta expected out on {device}, but got {out.device}"
        )
    if not (out.is_floating_point() or out.is_complex()):
        raise RuntimeError(
            f"result type Float can't be cast to the desired output type {out.dtype}"
        )
    if has_internal_overlapping(out) == MemOverlap.Yes:
        raise RuntimeError(
            "unsupported operation: more than one element of the written-to tensor "
            "refers to a single memory location"
        )
    aliases_input = False
    for tensor in inputs:
        if not _tensors_overlap(out, tensor):
            continue
        if not _is_exact_alias(out, tensor):
            raise RuntimeError(
                "unsupported operation: some elements of the input tensor and the "
                "written-to tensor refer to a single memory location"
            )
        aliases_input = True
    if tuple(out.shape) != tuple(shape):
        if aliases_input:
            raise RuntimeError(
                "special_zeta cannot resize an output that aliases an input"
            )
        out.resize_(shape)


def _fill_padded(value, is_tensor, shape, n, total, device):
    """Build an fp32 buffer of length ``total`` (a whole number of tiles).

    Valid lanes hold the broadcast/materialized operand; padding lanes hold an
    in-domain constant so the unmasked kernel never touches invalid math.
    """
    buf = torch.empty(total, dtype=torch.float32, device=device)
    if is_tensor:
        src = value.to(torch.float32)
        if tuple(src.shape) != tuple(shape):
            src = src.broadcast_to(shape)
        src = src.contiguous().reshape(-1)
        buf[:n].copy_(src)
        if total > n:
            buf[n:].fill_(_PAD_VALUE)
    else:
        buf.fill_(float(value))
    return buf


def _run(x, q, shape, device, dtype, *, x_is_tensor, q_is_tensor):
    n = 1
    for dim in shape:
        n *= dim
    result = torch.empty(tuple(shape), device=device, dtype=dtype)
    if n == 0:
        return result
    n_tiles = triton.cdiv(n, _BLOCK)
    total = n_tiles * _BLOCK
    xp = _fill_padded(x, x_is_tensor, shape, n, total, device)
    qp = _fill_padded(q, q_is_tensor, shape, n, total, device)
    out_buf = torch.empty(total, dtype=torch.float32, device=device)
    grid = (min(n_tiles, _MAX_GRID),)
    with runtime.torch_device_fn.device(device):
        _special_zeta_flat_kernel[grid](
            xp, qp, out_buf, n_tiles, BLOCK=_BLOCK, num_warps=4
        )
    result.copy_(out_buf[:n].reshape(tuple(shape)))
    return result


def special_zeta(x, q):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA")
    device = _validate_tensor_devices(x, q)
    dtype = _promoted_dtype(x, q)
    shape = torch.broadcast_shapes(x.shape, q.shape)
    return _run(x, q, shape, device, dtype, x_is_tensor=True, q_is_tensor=True)


def special_zeta_out(x, q, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA_OUT")
    device = _validate_tensor_devices(x, q)
    dtype = _promoted_dtype(x, q)
    shape = torch.broadcast_shapes(x.shape, q.shape)
    _prepare_out(out, shape, device, (x, q))
    result = _run(x, q, shape, device, dtype, x_is_tensor=True, q_is_tensor=True)
    out.copy_(result)
    return out


def special_zeta_tensor_scalar(x, q):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA_TENSOR_SCALAR")
    dtype = _promoted_dtype(x, q)
    return _run(x, q, x.shape, x.device, dtype, x_is_tensor=True, q_is_tensor=False)


def special_zeta_tensor_scalar_out(x, q, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA_TENSOR_SCALAR_OUT")
    dtype = _promoted_dtype(x, q)
    _prepare_out(out, x.shape, x.device, (x,))
    result = _run(x, q, x.shape, x.device, dtype, x_is_tensor=True, q_is_tensor=False)
    out.copy_(result)
    return out


def special_zeta_scalar_tensor(x, q):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA_SCALAR_TENSOR")
    dtype = _promoted_dtype(x, q)
    return _run(x, q, q.shape, q.device, dtype, x_is_tensor=False, q_is_tensor=True)


def special_zeta_scalar_tensor_out(x, q, out):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_ZETA_SCALAR_TENSOR_OUT")
    dtype = _promoted_dtype(x, q)
    _prepare_out(out, q.shape, q.device, (q,))
    result = _run(x, q, q.shape, q.device, dtype, x_is_tensor=False, q_is_tensor=True)
    out.copy_(result)
    return out
