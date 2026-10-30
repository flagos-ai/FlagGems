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

"""Independent numerical references and deterministic synthetic fixtures."""

import torch


def fp8_inputs(m, n, k, seed=41):
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = (torch.randn(m, k, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    b = (torch.randn(n, k, device="cuda", generator=g) * 0.25).to(torch.float8_e4m3fn)
    sa = torch.exp2(
        torch.randint(-3, 3, (m, k // 32), device="cuda", generator=g).float()
    )
    sb = torch.exp2(
        torch.randint(
            -3, 3, ((n + 31) // 32, k // 32), device="cuda", generator=g
        ).float()
    )
    return a, b, sa, sb


def fp8_reference(a, b, sa, sb, output_dtype=torch.bfloat16, **_):
    m, k = a.shape
    n = b.shape[0]
    out = torch.zeros((m, n), dtype=torch.float32, device=a.device)
    for i in range(k // 32):
        sl = slice(32 * i, 32 * (i + 1))
        dot = a[:, sl].float() @ b[:, sl].float().T
        out += dot * sa[:, i, None] * sb[:, i].repeat_interleave(32)[:n][None, :]
    return out.to(output_dtype)


def assert_exact(a, b):
    if isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a.shape == b.shape and a.dtype == b.dtype
        torch.testing.assert_close(a, b, rtol=0, atol=0, equal_nan=True)
        assert torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        ), "visible storage bytes differ"


def changed_graph(fn, args, refresh, reference, repeats=3):
    for _ in range(3):
        fn(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(*args)
    for seed in range(repeats):
        refresh(args, seed)
        graph.replay()
        torch.cuda.synchronize()
        assert_exact(out, reference(*args))


def native_block_fp8_baseline(A, B, As, Bs, output_dtype=torch.bfloat16):
    """Run the unchanged native schedule for exact-regression comparisons."""
    import triton

    from flag_gems.ops._w8a8_block_fp8_configs import (
        _BASE_CONFIGS,
        _H100_MIDRANGE_FP8_CONFIGS,
    )
    from flag_gems.ops._w8a8_block_fp8_sm90 import (
        _reduce_block_fp8_split_k,
        _w8a8_block_fp8_matmul,
        _w8a8_block_fp8_matmul_hopper,
    )
    from flag_gems.ops.w8a8_block_fp8_matmul import w8a8_block_fp8_matmul

    if not (A.is_cuda and A.device == B.device == As.device == Bs.device):
        raise ValueError("all inputs must share a CUDA device")
    if torch.cuda.get_device_capability(A.device) != (9, 0):
        raise ValueError("this implementation is validated on SM90 only")
    if A.dtype != torch.float8_e4m3fn or B.dtype != A.dtype:
        raise TypeError("A and B must use E4M3FN storage")
    if As.dtype != torch.float32 or Bs.dtype != torch.float32:
        raise TypeError("scales must be numeric FP32, not UE8M0 storage")
    if output_dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("output must be BF16 or FP32")
    if A.ndim != 2 or B.ndim != 2 or not A.is_contiguous() or not B.is_contiguous():
        raise ValueError("A and B must be contiguous matrices")
    m, k = A.shape
    n, bk = B.shape
    if (m, n, k) not in _H100_MIDRANGE_FP8_CONFIGS:
        return w8a8_block_fp8_matmul(A, B, As, Bs, [32, 32], output_dtype)
    if k <= 0 or k % 32 or bk != k or n <= 0:
        raise ValueError("K must be a positive multiple of 32 and match B")
    if As.shape != (m, k // 32) or Bs.shape != (triton.cdiv(n, 32), k // 32):
        raise ValueError("invalid block32 scale shapes")
    out = torch.empty((m, n), device=A.device, dtype=output_dtype)
    if m == 0:
        return out
    raw = _BASE_CONFIGS.get((n, k))
    config = (
        raw[min(raw, key=lambda x: abs(x - m))]
        if raw
        else dict(
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=32,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=32,
            num_warps=4,
            num_stages=3,
        )
    )
    hopper = bool(config.get("SWAP_AB", False) or config.get("SPLIT_K", 1) > 1)
    splits = config.get("SPLIT_K", 1) if hopper else 1
    kernel = _w8a8_block_fp8_matmul_hopper if hopper else _w8a8_block_fp8_matmul
    parts = (
        torch.empty((splits, m, n), device=A.device, dtype=torch.float32)
        if splits > 1
        else out
    )
    blocks = triton.cdiv(m, config["BLOCK_SIZE_M"]) * triton.cdiv(
        n, config["BLOCK_SIZE_N"]
    )
    kernel[(blocks, splits) if hopper else (blocks,)](
        A,
        B,
        parts,
        As,
        Bs,
        m,
        n,
        k,
        32,
        32,
        A.stride(0),
        A.stride(1),
        B.stride(1),
        B.stride(0),
        out.stride(0),
        out.stride(1),
        As.stride(0),
        As.stride(1),
        Bs.stride(1),
        Bs.stride(0),
        **config,
        needs_masking=bool(k % config["BLOCK_SIZE_K"]),
    )
    if splits > 1:
        _reduce_block_fp8_split_k[(triton.cdiv(m * n, 256),)](
            parts, out, m * n, splits, 256
        )
    return out
