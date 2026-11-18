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

"""special_ndtri: elementwise inverse standard-normal CDF.

The graded reference is torch.special.ndtri, whose device implementation is
ATen's Cephes `calc_ndtri`.  The kernel below reproduces that exact poly
algorithm (same constants, same Horner order, same two tail rationals and the
same branch/limit structure) in the element dtype, so the only deviation from
the reference is log/sqrt/divide rounding and FMA contraction (few ulp).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_gems.ops.special_ndtri import special_ndtri as default_special_ndtri
from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(
    f'flag_gems.runtime.backend._mthreads.ops.{__name__.split(".")[-1]}'
)

# The MUSA kernel computes in the element dtype, so only the dtypes the
# hardware can execute natively are whitelisted.  float16/bfloat16 are absent
# on purpose: the ATen reference (ndtri_cuda/ndtri_cpu) raises
# NotImplementedError for them, so there is no reference to compare
# against — let them fall back to the generic op, which reproduces that
# same rejection semantics without a dead branch inside the kernel.
_SUPPORTED_DTYPES = {torch.float32}

_MAX_RANK = 8

_BLOCK = 1024
_WARPS = 4

# Bounded launch-configuration search for the streaming (contiguous) path.
# (1024, 4) -- the round-2 fixed launch -- is included so that a flat spread is
# interpretable.  Triton appends every argument dtype to the autotune key, so
# float32 and float64 tune independently.
_CONFIGS = [
    triton.Config({"BLOCK": 256}, num_warps=2),
    triton.Config({"BLOCK": 256}, num_warps=4),
    triton.Config({"BLOCK": 512}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=4),
    triton.Config({"BLOCK": 1024}, num_warps=8),
    triton.Config({"BLOCK": 2048}, num_warps=8),
]


def _size_bucket(n):
    """Coarse numel bucket so autotune does not re-tune for every new shape."""
    if n <= 1 << 13:
        return 0
    if n <= 1 << 22:
        return 1
    return 2


@triton.jit
def _ndtri_of(y0):
    """Cephes calc_ndtri, evaluated branchlessly in the element dtype."""
    s2pi = 2.50662827463100050242
    # exp(-2); also used as ATen's 1.0 - exp(-2) split point.
    ec2 = 0.13533528323661269189

    upper = y0 > (1.0 - ec2)
    y = tl.where(upper, 1.0 - y0, y0)
    central_mask = y > ec2

    # ---- central rational approximation, 0 <= |y - 0.5| <= 3/8 ------------
    yc = y - 0.5
    y2 = yc * yc
    p0 = (
        (
            (-5.99633501014107895267e1 * y2 + 9.80010754185999661536e1) * y2
            - 5.66762857469070293439e1
        )
        * y2
        + 1.39312609387279679503e1
    ) * y2 - 1.23916583867381258016
    q0 = (
        (
            (
                (
                    (
                        ((y2 + 1.95448858338141759834) * y2 + 4.67627912898881538453)
                        * y2
                        + 8.63602421390890590575e1
                    )
                    * y2
                    - 2.25462687854119370527e2
                )
                * y2
                + 2.00260212380060660359e2
            )
            * y2
            - 8.20372256168333339912e1
        )
        * y2
        + 1.59056225126211695515e1
    ) * y2 - 1.18331621121330003142
    central = (yc + yc * (y2 * p0 / q0)) * s2pi

    # ---- tail, z = 1/sqrt(-2 log y) ---------------------------------------
    t = tl.sqrt(-2.0 * tl.log(y))
    x0 = t - tl.log(t) / t
    z = 1.0 / t

    p1 = (
        (
            (
                (
                    (
                        (
                            (4.05544892305962419923 * z + 3.15251094599893866154e1) * z
                            + 5.71628192246421288162e1
                        )
                        * z
                        + 4.40805073893200834700e1
                    )
                    * z
                    + 1.46849561928858024014e1
                )
                * z
                + 2.18663306850790267539
            )
            * z
            - 1.40256079171354495875e-1
        )
        * z
        - 3.50424626827848203418e-2
    ) * z - 8.57456785154685413611e-4
    q1 = (
        (
            (
                (
                    (
                        ((z + 1.57799883256466749731e1) * z + 4.53907635128879210584e1)
                        * z
                        + 4.13172038254672030440e1
                    )
                    * z
                    + 1.50425385692907503408e1
                )
                * z
                + 2.50464946208309415979
            )
            * z
            - 1.42182922854787788574e-1
        )
        * z
        - 3.80806407691578277194e-2
    ) * z - 9.33259480895457427372e-4
    p2 = (
        (
            (
                (
                    (
                        (
                            (3.23774891776946035970 * z + 6.91522889068984211695) * z
                            + 3.93881025292474443415
                        )
                        * z
                        + 1.33303460815807542389
                    )
                    * z
                    + 2.01485389549179081538e-1
                )
                * z
                + 1.23716634817820021358e-2
            )
            * z
            + 3.01581553508235416007e-4
        )
        * z
        + 2.65806974686737550832e-6
    ) * z + 6.23974539184983293730e-9
    q2 = (
        (
            (
                (
                    (
                        ((z + 6.02427039364742014255) * z + 3.67983563856160859403) * z
                        + 1.37702099489081330271
                    )
                    * z
                    + 2.16236993594496635890e-1
                )
                * z
                + 1.34204006088543189037e-2
            )
            * z
            + 3.28014464682127739104e-4
        )
        * z
        + 2.89247864745380683936e-6
    ) * z + 6.79019408009981274425e-9
    near = t < 8.0
    num = tl.where(near, p1, p2)
    den = tl.where(near, q1, q2)
    tail = x0 - z * num / den
    # ATen negates only for the lower tail (y0 <= 1 - exp(-2)).
    tail = tl.where(upper, tail, -tail)

    out = tl.where(central_mask, central, tail)

    # Boundaries and out-of-domain, exactly as ATen short-circuits them.
    out = tl.where(y0 == 0.0, -float("inf"), out)
    out = tl.where(y0 == 1.0, float("inf"), out)
    out = tl.where((y0 < 0.0) | (y0 > 1.0), float("nan"), out)
    return out


@triton.autotune(configs=_CONFIGS, key=["size_bucket"])
@triton.jit
def _ndtri_contig_kernel(x_ptr, out_ptr, n_elements, size_bucket, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements
    v = tl.load(x_ptr + offs, mask=mask, other=1.0)
    tl.store(out_ptr + offs, _ndtri_of(v), mask=mask)


@triton.jit
def _ndtri_strided_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,
    t0,
    t1,
    t2,
    t3,
    t4,
    t5,
    t6,
    t7,
    NDIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
    mask = offs < n_elements

    idx = offs
    src = tl.zeros((BLOCK,), dtype=tl.int64)
    if NDIM > 7:
        src += (idx % s7) * t7
        idx = idx // s7
    if NDIM > 6:
        src += (idx % s6) * t6
        idx = idx // s6
    if NDIM > 5:
        src += (idx % s5) * t5
        idx = idx // s5
    if NDIM > 4:
        src += (idx % s4) * t4
        idx = idx // s4
    if NDIM > 3:
        src += (idx % s3) * t3
        idx = idx // s3
    if NDIM > 2:
        src += (idx % s2) * t2
        idx = idx // s2
    if NDIM > 1:
        src += (idx % s1) * t1
        idx = idx // s1
    src += idx * t0

    v = tl.load(x_ptr + src, mask=mask, other=1.0)
    tl.store(out_ptr + offs, _ndtri_of(v), mask=mask)


def _collapse(shape, strides):
    """Drop size-1 dims and merge neighbouring dims contiguous in memory."""
    dims = [(int(s), int(st)) for s, st in zip(shape, strides) if int(s) != 1]
    merged = []
    for s, st in dims:
        if merged:
            ps, pst = merged[-1]
            if pst == s * st:
                merged[-1] = (ps * s, st)
                continue
        merged.append((s, st))
    if not merged:
        return (1,), (1,)
    return tuple(m[0] for m in merged), tuple(m[1] for m in merged)


def _launch_strided(x, out, shape, strides):
    n = out.numel()
    rank = len(shape)
    shp = tuple(shape) + (1,) * (_MAX_RANK - rank)
    strd = tuple(strides) + (0,) * (_MAX_RANK - rank)
    grid = (triton.cdiv(n, _BLOCK),)
    _ndtri_strided_kernel[grid](
        x,
        out,
        n,
        shp[0],
        shp[1],
        shp[2],
        shp[3],
        shp[4],
        shp[5],
        shp[6],
        shp[7],
        strd[0],
        strd[1],
        strd[2],
        strd[3],
        strd[4],
        strd[5],
        strd[6],
        strd[7],
        NDIM=rank,
        BLOCK=_BLOCK,
        num_warps=_WARPS,
    )


def _use_triton_kernel(x):
    """Whether ``x`` can be handled by the MUSA kernels."""
    if not isinstance(x, torch.Tensor):
        return False
    if x.device.type != "musa" or x.dtype not in _SUPPORTED_DTYPES:
        return False
    if x.is_complex():
        return False
    return True


def special_ndtri(self):
    logger.debug("GEMS_MTHREADS SPECIAL_NDTRI")
    if not _use_triton_kernel(self):
        return default_special_ndtri(self)
    with torch_device_fn.device(self.device):
        return _launch_ndtri(self)


def _launch_ndtri(self):
    x = self
    dtype = x.dtype
    if dtype is not torch.float32 and dtype is not torch.float64:
        raise NotImplementedError(f"\"ndtri_cuda\" not implemented for '{dtype}'")

    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    n = out.numel()
    if n == 0:
        return out

    if x.is_contiguous() or n == 1:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)  # noqa: E731
        _ndtri_contig_kernel[grid](x, out, n, _size_bucket(n))
        return out

    shape, strides = _collapse(x.shape, x.stride())
    if len(shape) <= _MAX_RANK:
        _launch_strided(x, out, shape, strides)
        return out

    # Extremely high-rank views: peel the leading dimension with zero-copy
    # narrows (each level removes one dimension, so this terminates).
    for i in range(shape[0]):
        _launch_strided(x.narrow(0, i, 1), out.narrow(0, i, 1), shape[1:], strides[1:])
    return out
