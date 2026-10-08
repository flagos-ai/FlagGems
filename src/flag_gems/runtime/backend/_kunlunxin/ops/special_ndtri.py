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
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig

from ..utils.pointwise_dynamic import pointwise_dynamic

# The logger name is deliberately the *generic* module name: tests/
# test_special_ndtri.py does `caplog.at_level("DEBUG", logger=
# "flag_gems.ops.special_ndtri")`, and a record emitted on
# `_kunlunxin.ops.special_ndtri` would inherit root's WARNING level and be
# dropped, breaking the `assert "GEMS SPECIAL_NDTRI" in caplog.text` in all four
# test functions.  `record.pathname` still points at this file, which is what the
# dispatch evidence uses.
logger = logging.getLogger("flag_gems.ops.special_ndtri")

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    buffer_size_limit=2048,
    isCloseVectorization=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)

# ndtri is implemented for single and double precision only in PyTorch
# (ndtri_cpu / ndtri_cuda both raise NotImplementedError for Half and BFloat16);
# the same gate is kept so this override matches the reference.  Note that on
# this device torch.float64 silently degrades to float32 (a tensor created with
# dtype=torch.float64 reports .dtype == torch.float32 and element_size() == 4),
# so the float64 entry only matters for callers that pass the dtype through.
_SUPPORTED_DTYPES = (torch.float32, torch.float64)


@pointwise_dynamic(promotion_methods=[(0, "DEFAULT")], config=config_)
@triton.jit
def special_ndtri_func(x):
    p = x.to(tl.float32)
    q = p - 0.5
    aq = tl.abs(q)

    # ----- central region: |q| <= 0.425  ->  q * A(r)/B(r), r = 0.180625 - q^2
    rc = 0.180625 - q * q
    num_c = 3.3871327179 + rc * (50.434271938 + rc * (159.29113202 + rc * 59.109374720))
    den_c = 1.0 + rc * (17.895169469 + rc * (78.757757664 + rc * 67.187563600))
    x_central = q * (num_c / den_c)

    # ----- tails: r = sqrt(-log(min(p, 1-p)))
    pt = tl.where(q < 0.0, p, 1.0 - p)
    r = tl.sqrt(-tl.log(pt))

    # r <= 5 branch; argument intentionally unclamped so nan survives
    r1 = r - 1.6
    num_1 = 1.4234372777 + r1 * (
        2.7568153900 + r1 * (1.3067284816 + r1 * 0.17023821103)
    )
    den_1 = 1.0 + r1 * (0.73700164250 + r1 * 0.12021132975)

    # r > 5 branch; r is clamped to 1e16 so that r == +inf yields +inf instead of
    # inf/inf = nan.  The clamp is arithmetic on purpose: 1/(1/inf + 1e-16) is
    # exactly 1e16, while for every finite r in [0.83, 9.4] the 1e-16 sits nine
    # orders of magnitude below an fp32 ulp of 1/r, so it is a no-op there.  A
    # `tl.minimum` would cost ~1.0 ms at [4096,4096] (measured) and would also
    # destroy nan; a `tl.where` costs ~0.29 ms; this costs ~0.11 ms.
    r2 = 1.0 / (1.0 / r + 1e-16) - 5.0
    num_2 = 6.6579051150 + r2 * (
        3.0812263860 + r2 * (0.42868294337 + r2 * 0.017337203997)
    )
    den_2 = 1.0 + r2 * (0.24197894225 + r2 * 0.012258202635)

    x_tail = (q / aq) * tl.where(r > 5.0, num_2 / den_2, num_1 / den_1)

    return tl.where(aq <= 0.425, x_central, x_tail).to(x.dtype)


def special_ndtri(self):
    logger.debug("GEMS_KUNLUNXIN SPECIAL_NDTRI")
    if self.dtype not in _SUPPORTED_DTYPES:
        raise NotImplementedError(
            f"\"ndtri\" not implemented for '{self.dtype}'; supported dtypes are "
            f"{_SUPPORTED_DTYPES}"
        )
    return special_ndtri_func(self)
