# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import subprocess
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - initialize NPU before importing Triton
import triton
import triton.language as tl
import triton.language.extra.cann.extension as al

from flag_gems.utils import triton_lang_extension as ext

from . import compile_fixpipe as helper

helper.install_cann90_custom_op_compat()
ROOT = Path(__file__).parent
src = ROOT / "vector_epilogue.cpp"
REV = hashlib.sha256(
    src.read_bytes() + Path(helper.__file__).read_bytes() + Path(__file__).read_bytes()
).hexdigest()
bc = ROOT / ("epilogue_brcb_ub_" + REV[:16] + ".bc")
if not bc.exists():
    subprocess.check_call(
        [
            s.replace("dav-c220-cube", "dav-c220-vec")
            for s in helper.compile_cmd(src, bc)
        ]
        + ["-O3"]
    )


@al.register_custom_op
class epilogue_brcb_ub:
    name = "epilogue_brcb_ub"
    core = al.CORE.VECTOR
    pipe = al.PIPE.PIPE_V
    mode = al.MODE.SIMD
    extra_attr = "flaggems_pass_outputs=true"
    symbol = "epilogue_brcb_ub"
    bitcode = str(bc)
    source = str(src)
    compile = (
        helper.makefile_compile()
        .replace("dav-c220-cube", "dav-c220-vec")
        .replace("-std=c++17", "-O3 -std=c++17")
    )

    def __init__(self, c, a, b, br, row, r, x, cached_a, out=None):
        assert out is not None
        for key in ("row", "r", "x", "cached_a"):
            self.arg_type[key] = tl.int32


@triton.jit
def scale_custom(
    C,
    A,
    B,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    NP: tl.constexpr,
    R: tl.constexpr,
    X: tl.constexpr,
    CORES: tl.constexpr,
    BF16: tl.constexpr,
    REVISION: tl.constexpr,
):
    with al.scope(core_mode="vector"):
        pid = ext.program_id(0)
        CACHE_A: tl.constexpr = M <= 256 and N >= 4096
        CACHE_B: tl.constexpr = CORES % (N // X) == 0
        if CACHE_A:
            aa = tl.arange(0, triton.next_power_of_2(M))
            a_all = tl.load(A + aa, aa < M, other=0)
        if CACHE_B:
            cc = pid % (N // X) * X + tl.arange(0, X)
            b_all = tl.load(B + cc)
        for tile in range(pid, (M // R) * (N // X), CORES):
            row = tile // (N // X) * R
            col = tile % (N // X) * X
            rr = row + tl.arange(0, R)
            cc = col + tl.arange(0, X)
            c = tl.load(C + rr[:, None] * NP + cc[None, :])
            if CACHE_A:
                a = a_all
            else:
                a = tl.load(A + rr)
            if CACHE_B:
                b = b_all
            else:
                b = tl.load(B + cc)
            br = tl.full([R * 8], 0, tl.float32)
            value = tl.full([R, X], 0, tl.float32)
            result = al.custom(
                "epilogue_brcb_ub",
                c,
                a,
                b,
                br,
                row,
                R,
                X,
                1 if CACHE_A else 0,
                out=value,
            )
            tl.store(O + rr[:, None] * N + cc[None, :], result)


def prepare(c, sa, sb, out, m, n, r=8, x=256):
    assert (
        c.dtype == torch.int32
        and sa.dtype == torch.float32
        and sb.dtype == torch.float32
    )
    assert out.dtype in (torch.bfloat16, torch.float16) and out.is_contiguous()
    assert (
        m > 0 and n > 0 and c.stride(1) == 1 and sa.stride(0) == 1 and sb.stride(0) == 1
    )
    assert (
        m % r == 0
        and n % x == 0
        and r in (8, 16, 32)
        and x in (64, 128, 256, 512, 1024)
        and r * x <= 8192
    )
    cores = min(40, m // r * (n // x))
    args = (
        c,
        sa,
        sb,
        out,
        m,
        n,
        c.stride(0),
        r,
        x,
        cores,
        out.dtype == torch.bfloat16,
        REV,
    )
    return lambda: scale_custom[(cores,)](*args), args, cores
