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

import torch

from flag_gems.fused.mhc.mhc_bwd import mhc_bwd as _general_mhc_bwd
from flag_gems.ops import pad as _gems_pad

_BLOCK_S = 64  # must match BLOCK_S in the general _mhc_bwd_kernel_n4


def mhc_bwd(
    out: torch.Tensor,
    dout: torch.Tensor,
    cg_iters: int = None,
) -> torch.Tensor:
    """Sinkhorn backward (kunlunxin / XPU) with XPU-safe input guards.

    Same interface and semantics as `flag_gems.fused.mhc.mhc_bwd.mhc_bwd`;
    delegates to the general implementation for shapes that are safe on XPU.
    """
    if out.numel() == 0:
        return torch.empty(out.shape, dtype=torch.float32, device=out.device)

    seqlen = out.shape[0]
    if seqlen % _BLOCK_S != 0:
        pad = (-seqlen) % _BLOCK_S
        out_p = _gems_pad(out, (0, 0, 0, 0, 0, pad))
        dout_p = _gems_pad(dout, (0, 0, 0, 0, 0, pad))
        res = _general_mhc_bwd(out_p, dout_p, cg_iters=cg_iters)
        return res[:seqlen]

    return _general_mhc_bwd(out, dout, cg_iters=cg_iters)


def _install():
    """Wire the XPU entry into the direct-import entrypoint.

    The mhc fused family is called via direct module import
    (`from flag_gems.fused.mhc.mhc_bwd import mhc_bwd`) in both
    tests/test_mhc_ops.py and benchmark/test_mhc.py, so the normal
    SpecOpRegistrar namespace swap can not reach it. Replace the attribute on
    the already-imported module (loaded during `import flag_gems`).
    """
    import sys

    mod = sys.modules.get("flag_gems.fused.mhc.mhc_bwd")
    if mod is not None:
        cur = getattr(mod, "mhc_bwd", None)
        if cur is _general_mhc_bwd:
            mod.mhc_bwd = mhc_bwd


_install()
