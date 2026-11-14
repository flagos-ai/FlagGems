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
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry

from ..utils.pointwise_dynamic import pointwise_dynamic

logger = logging.getLogger(__name__)

# Minimum element count to use the tle.raw payload (run_tests 1M/16.7M shapes
# take the payload; small shapes keep the pointwise path to avoid launch floor).
_PAYLOAD_NUMEL_MIN = 1_000_000

_config = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=4096,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=16,
)


@pointwise_dynamic(
    is_tensor=[True, False, False],
    promotion_methods=[(0, "INT_TO_FLOAT")],
    num_outputs=1,
    config=_config,
)
@triton.jit
def dequantize_func(inp, zero_point, scale):
    return (inp - zero_point) * scale


# ===== tle.raw payload (xpu3 cluster path): bypass i8->f32 scalarization / 32B
# ===== DMA / missing pipeline =====
# Precompiled artifact: payload/obj/dequantize_raw.o (built from
# payload/src/dequantize_raw.xpu with xpu-clang; source is not committed, see
# payload/.gitignore; clear ~/.triton/cache after recompiling).
# object= performs a relocatable link (no inlining), matching gems_quantize.
_PAY_OBJ = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "payload", "obj"
)
try:
    import triton.experimental.tle as tle

    @tle.raw.dialect("xpu3", object=os.path.join(_PAY_OBJ, "dequantize_raw.o"), arch=3)
    def dequantize_raw(in_ptr, out_ptr, n, scale, zero_point, start, count): ...

    @libentry()
    @triton.jit(do_not_specialize=["n", "scale", "zero_point", "start", "count", "per"])
    def _dequantize_raw_kernel(In, Out, n, scale, zero_point, start, count, per):
        pid = tl.program_id(0)
        tle.raw.call(
            dequantize_raw,
            (
                In,
                Out,
                tl.cast(n, tl.int64),
                scale,
                tl.cast(zero_point, tl.int32),
                tl.cast(start + pid * per, tl.int64),
                tl.cast(per, tl.int64),
            ),
        )

    _HAS_TLE_RAW = True
except Exception:  # pragma: no cover - tle/file= unavailable
    _HAS_TLE_RAW = False


def _dequantize_payload(raw, out, n, scale, zero_point):
    """Host driver: slice by cluster; the 64 cores of a cluster split by chunk."""
    nclusters = min(12, max((n + 64 * 512 - 1) // (64 * 512), 1))
    per = ((n + nclusters - 1) // nclusters + 511) // 512 * 512  # 512-aligned
    grid = (triton.cdiv(n, per), 1, 1)
    with torch_device_fn.device(raw.device):
        _dequantize_raw_kernel[grid](raw, out, n, scale, zero_point, 0, per, per)


def dequantize(a):
    """Dequantize a quantized tensor (qint8/quint8/qint32) to float32 on XPU."""
    logger.debug("GEMS_KUNLUNXIN DEQUANTIZE")

    scale = float(a.q_scale())
    zero_point = int(a.q_zero_point())

    if a.numel() == 0:
        return torch.empty(a.shape, dtype=torch.float32, device=a.device)

    # `a.int_repr()` has no kernel in the XPU build; use a zero-copy int8 view
    # of the quantized storage instead. The arithmetic stays in the kernel.
    raw = torch.empty(0, dtype=torch.int8, device=a.device)
    raw.set_(a.untyped_storage(), a.storage_offset(), a.size(), a.stride())
    if not raw.is_contiguous():
        raw = raw.contiguous()

    if _HAS_TLE_RAW and raw.numel() >= _PAYLOAD_NUMEL_MIN:
        out = torch.empty(a.shape, dtype=torch.float32, device=a.device)
        _dequantize_payload(raw, out, raw.numel(), scale, zero_point)
        return out

    return dequantize_func(raw, zero_point, scale)
