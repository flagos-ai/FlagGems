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

"""Multi-GPU test of the fused RS + add + RMSNorm + AG, its RS-only mode (gather=False)
and the q/kv norm + AG kernel. Needs torchrun:

    torchrun --nproc_per_node 4 tests/test_rs_norm_ag.py [--bench]
"""

import argparse
import os

import pytest
import torch
import torch.distributed as dist

try:
    from flag_gems.fused.rs_norm_ag import (
        fused_qkv_norm_ag,
        fused_rs_add_rms_norm_ag,
        ppu810e_rings,
        row_range,
    )
except ImportError:  # run against an installed flag_gems without this file
    import importlib.util

    _path = os.path.join(
        os.path.dirname(__file__), "..", "src", "flag_gems", "fused", "rs_norm_ag.py"
    )
    _spec = importlib.util.spec_from_file_location("rs_norm_ag", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    fused_qkv_norm_ag, fused_rs_add_rms_norm_ag = _mod.fused_qkv_norm_ag, _mod.fused_rs_add_rms_norm_ag
    ppu810e_rings, row_range = _mod.ppu810e_rings, _mod.row_range

H = 6144
EPS = 1e-6
# GLM-5 / DeepSeek MLA qkv_a output: q_lora 2048 | kv_lora 512 | k_pe 64
Q_DIM, KV_DIM, ROW_DIM = 2048, 512, 2624


def _inputs(seed, rank, T, device, dtype):
    g = torch.Generator(device=device).manual_seed(seed * 1000 + rank)
    return torch.randn(T, H, generator=g, device=device, dtype=torch.float32).to(dtype)


def _reference(seed, T, W, device, dtype, res_full, weight):
    total = torch.zeros(T, H, device=device, dtype=torch.float32)
    for r in range(W):
        total += _inputs(seed, r, T, device, dtype).float()
    x = total + res_full.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * weight.float()
    return y.to(dtype), x.to(dtype)


def check(comm, rank, W, device, dtype):
    torch.manual_seed(0)
    weight = torch.randn(H, device=device, dtype=dtype)
    cases = [1, 3, W - 1, W, W + 1, 7, 100, 1000, 4095, 4096, 1, 4096]
    # the ring sums partials in bf16 hop by hop (like pccl), the reference in fp32
    atol, rtol = 1e-1, 1.6e-2
    # per case: [max |out-ref|, #out beyond tol, max |res-ref|, #res beyond tol,
    #            ranks differ / timeouts, max |res-ref| of all_reduce (pccl, bf16) + add]
    stats = torch.zeros(len(cases), 6, device=device)
    for i, T in enumerate(cases):
        seed = 17 + i
        g = torch.Generator(device=device).manual_seed(seed)
        res_full = torch.randn(T, H, generator=g, device=device).to(dtype)
        b, e = row_range(T, W, rank)
        x = _inputs(seed, rank, T, device, dtype)
        residual = res_full[b:e].clone()

        # what runs today: all_reduce in bf16, then the fp32 residual add of the norm
        ar = x.clone()
        dist.all_reduce(ar)
        base_res = (ar.float() + res_full.float()).to(dtype)

        out = comm(x, residual, weight, EPS)
        torch.cuda.synchronize(device)

        ref_out, ref_res = _reference(seed, T, W, device, dtype, res_full, weight)
        d_out = (out.float() - ref_out.float()).abs()
        d_res = (residual.float() - ref_res[b:e].float()).abs()
        stats[i, 0] = d_out.max()
        stats[i, 1] = (d_out > atol + rtol * ref_out.float().abs()).sum()
        stats[i, 5] = (base_res.float() - ref_res.float()).abs().max()
        if d_res.numel():
            stats[i, 2] = d_res.max()
            # many-hop bf16 sums: the residual only has to be as close as all_reduce's
            stats[i, 3] = (d_res > torch.clamp(2 * stats[i, 5], min=atol)).sum()
        # every rank must hold the same bits; every rank runs this collective, so a
        # failing rank never leaves the others waiting
        gathered = [torch.empty_like(out) for _ in range(W)]
        dist.all_gather(gathered, out)
        stats[i, 4] = float(not all(torch.equal(gathered[0], o) for o in gathered))

    stats[:, 4] += comm.err.float()
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    bad = (stats[:, 1] > 0) | (stats[:, 3] > 0) | (stats[:, 4] > 0)
    if rank == 0:
        for i, T in enumerate(cases):
            s = stats[i].tolist()
            flag = "FAIL" if bad[i] else "ok"
            print(
                f"  T={T:5d} {flag:4s} out max {s[0]:.4g} (#bad {int(s[1])})  "
                f"res max {s[2]:.4g} (#bad {int(s[3])}, all_reduce {s[5]:.4g})  "
                f"differ/timeouts {int(s[4])}"
            )
    assert not bad.any().item(), "rs_norm_ag mismatch, see the table above"
    if rank == 0:
        print(f"check ok: {len(cases)} calls, max |out - ref| = {stats[:, 0].max().item():.4g}")
        if os.environ.get("RS_NORM_AG_DUMP_ASM"):
            import collections
            import re

            llir = comm.last_kernel.asm.get("llir", "")
            ops = collections.Counter(re.findall(r"ppu\.(?:ld|st|atom)[\w.]*", llir))
            print("memory ops in llir:", dict(ops))


def check_rs_only(comm, rank, W, device, dtype):
    """gather=False (RS + add + norm, rows stay scattered), interleaved with gather=True
    calls so the flag bases of both kinds of call stay in step."""
    torch.manual_seed(1)
    weight = torch.randn(H, device=device, dtype=dtype)
    cases = [(1, False), (W, False), (W + 1, True), (100, False), (4095, False), (7, True), (4096, False)]
    atol, rtol = 1e-1, 1.6e-2
    # per case: [#out beyond tol, #res beyond tol, timeouts]
    stats = torch.zeros(len(cases), 3, device=device)
    for i, (T, gather) in enumerate(cases):
        seed = 57 + i
        g = torch.Generator(device=device).manual_seed(seed)
        res_full = torch.randn(T, H, generator=g, device=device).to(dtype)
        b, e = row_range(T, W, rank)
        x = _inputs(seed, rank, T, device, dtype)
        residual = res_full[b:e].clone()
        out = comm(x, residual, weight, EPS, gather=gather)
        torch.cuda.synchronize(device)
        ref_out, ref_res = _reference(seed, T, W, device, dtype, res_full, weight)
        if gather:
            ref_mine = ref_out
        else:
            ref_mine = ref_out[b:e]
            assert out.shape == (e - b, H) and out.data_ptr() == x[b:e].data_ptr()
        stats[i, 0] = ((out.float() - ref_mine.float()).abs() > atol + rtol * ref_mine.float().abs()).sum()
        # residual: same bound as check() uses when all_reduce stays inside it
        stats[i, 1] = ((residual.float() - ref_res[b:e].float()).abs() > 0.5).sum()
    stats[:, 2] += comm.err.float()
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    bad = stats.sum(1) > 0
    if rank == 0:
        for i, (T, gather) in enumerate(cases):
            s = stats[i].tolist()
            print(
                f"  rs-only T={T:5d} gather={int(gather)} {'FAIL' if bad[i] else 'ok'}  "
                f"#bad out {int(s[0])} res {int(s[1])} timeouts {int(s[2])}"
            )
    assert not bad.any().item(), "gather=False mismatch, see the table above"
    if rank == 0:
        print(f"rs-only check ok: {len(cases)} calls")


def _qkv_reference(xf, qw, kvw):
    x = xf.float()
    q, kv, rest = x[:, :Q_DIM], x[:, Q_DIM : Q_DIM + KV_DIM], x[:, Q_DIM + KV_DIM :]
    q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + EPS) * qw.float()
    kv = kv * torch.rsqrt(kv.pow(2).mean(-1, keepdim=True) + EPS) * kvw.float()
    return torch.cat([q, kv, rest], -1).to(xf.dtype)


def check_qkv(qcomm, rank, W, device, dtype):
    torch.manual_seed(2)
    qw = torch.randn(Q_DIM, device=device, dtype=dtype)
    kvw = torch.randn(KV_DIM, device=device, dtype=dtype)
    cases = [1, 3, W - 1, W, W + 1, 7, 100, 1000, 4095, 4096, 1, 4096]
    # bf16 out of the same fp32 math: 1 ulp of rounding-order slack
    atol, rtol = 2e-2, 1.6e-2
    # per case: [max |out-ref|, #beyond tol, rest not bit-exact, ranks differ / timeouts]
    stats = torch.zeros(len(cases), 4, device=device)
    for i, T in enumerate(cases):
        g = torch.Generator(device=device).manual_seed(91 + i)
        xf = torch.randn(T, ROW_DIM, generator=g, device=device).to(dtype)
        b, e = row_range(T, W, rank)
        out = qcomm(xf[b:e].clone(), T, qw, EPS, kvw, EPS)
        torch.cuda.synchronize(device)
        ref = _qkv_reference(xf, qw, kvw)
        d = (out.float() - ref.float()).abs()
        stats[i, 0] = d.max()
        stats[i, 1] = (d > atol + rtol * ref.float().abs()).sum()
        stats[i, 2] = float(not torch.equal(out[:, Q_DIM + KV_DIM :], xf[:, Q_DIM + KV_DIM :]))
        gathered = [torch.empty_like(out) for _ in range(W)]
        dist.all_gather(gathered, out)
        stats[i, 3] = float(not all(torch.equal(gathered[0], o) for o in gathered))
    stats[:, 3] += qcomm.err.float()
    dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    bad = (stats[:, 1:] > 0).any(1)
    if rank == 0:
        for i, T in enumerate(cases):
            s = stats[i].tolist()
            print(
                f"  qkv T={T:5d} {'FAIL' if bad[i] else 'ok':4s} max {s[0]:.4g} (#bad {int(s[1])})  "
                f"rest not exact {int(s[2])}  differ/timeouts {int(s[3])}"
            )
    assert not bad.any().item(), "qkv_norm_ag mismatch, see the table above"
    if rank == 0:
        print(f"qkv check ok: {len(cases)} calls, max |out - ref| = {stats[:, 0].max().item():.4g}")


def bench_split(comm, qcomm, rank, W, device, dtype, tokens, iters):
    """Comm + norm of MoE output -> next layer's attention input (qkv_a GEMM left out,
    the fused kernels do not contain it):
    native  AR + residual add + input norm (T rows) ... q norm + kv norm (T rows)
    fused   fuse2 (RS + add + norm)            ... fuse3 (q / kv norm + AG)"""
    import flag_gems

    weight = torch.randn(H, device=device, dtype=dtype)
    qw = torch.randn(Q_DIM, device=device, dtype=dtype)
    kvw = torch.randn(KV_DIM, device=device, dtype=dtype)
    if rank == 0:
        print(
            f"{'T':>6} {'native':>9} {'fused':>9} {'speedup':>8} | "
            f"{'AR+norm':>9} {'2norm':>9} {'fuse2':>9} {'fuse3':>9}  (us per call)"
        )
    for T in tokens:
        b, e = row_range(T, W, rank)
        x0 = torch.randn(T, H, device=device, dtype=dtype)
        r0 = torch.randn(T, H, device=device, dtype=dtype)
        x, res = x0.clone(), r0.clone()
        xf, rloc = x0.clone(), r0[b:e].clone()
        # qkv_a output: every row on native, this rank's rows on fused
        qkv_full = torch.randn(T, ROW_DIM, device=device, dtype=dtype)
        qkv_loc = qkv_full[b:e].clone()
        qout = torch.empty(T, ROW_DIM, device=device, dtype=dtype)

        def reset_native():
            x.copy_(x0)
            res.copy_(r0)

        def ar_norm():
            # x ends up normed in place, so it stays bounded across calls
            dist.all_reduce(x)
            flag_gems.fused_add_rms_norm(x, res, [H], weight, EPS)

        def qkv_norm():
            flag_gems.rms_norm(qkv_full[:, :Q_DIM], [Q_DIM], qw, EPS)
            flag_gems.rms_norm(qkv_full[:, Q_DIM : Q_DIM + KV_DIM], [KV_DIM], kvw, EPS)

        def native():
            ar_norm()
            qkv_norm()

        def reset_fused():
            xf.copy_(x0)
            rloc.copy_(r0[b:e])

        def fuse2():
            comm(xf, rloc, weight, EPS, gather=False)

        def fuse3():
            qcomm(qkv_loc, T, qw, EPS, kvw, EPS, out=qout)

        def fused():
            fuse2()
            fuse3()

        t_nat = _time(native, reset_native, iters, device)
        t_fu = _time(fused, reset_fused, iters, device)
        t_ar = _time(ar_norm, reset_native, iters, device)
        t_qn = _time(qkv_norm, lambda: None, iters, device)
        t_f2 = _time(fuse2, reset_fused, iters, device)
        t_f3 = _time(fuse3, lambda: None, iters, device)
        if rank == 0:
            print(
                f"{T:>6} {t_nat:>9.1f} {t_fu:>9.1f} {t_nat / t_fu:>7.2f}x | "
                f"{t_ar:>9.1f} {t_qn:>9.1f} {t_f2:>9.1f} {t_f3:>9.1f}"
            )
    assert comm.errors() == 0 and qcomm.errors() == 0, "barrier timeouts"


def compare_baseline(comm, rank, W, device, dtype):
    """Against what runs today: all_reduce + flag_gems fused_add_rms_norm."""
    import flag_gems

    T = 4096
    weight = torch.randn(H, device=device, dtype=dtype)
    x = _inputs(99, rank, T, device, dtype)
    res_full = torch.randn(T, H, device=device, dtype=dtype)
    b, e = row_range(T, W, rank)

    xa, ra = x.clone(), res_full.clone()
    dist.all_reduce(xa)
    flag_gems.fused_add_rms_norm(xa, ra, [H], weight, EPS)

    xf, rf = x.clone(), res_full[b:e].clone()
    comm(xf, rf, weight, EPS)
    torch.cuda.synchronize(device)
    d_out = (xf.float() - xa.float()).abs().max().item()
    d_res = (rf.float() - ra[b:e].float()).abs().max().item()
    if rank == 0:
        print(f"vs all_reduce + flag_gems: max |out| diff {d_out:.4g}, |residual| diff {d_res:.4g}")


def _time(fn, reset, iters, device, rounds=5):
    # back-to-back calls amortize the launch skew between ranks; every fn keeps its
    # values bounded without a reset (normed output written back, xp read only)
    reset()
    for _ in range(3):
        fn()
    per_call = []
    for _ in range(rounds):
        dist.barrier()
        torch.cuda.synchronize(device)
        s, t = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        t.record()
        torch.cuda.synchronize(device)
        per_call.append(s.elapsed_time(t) / iters * 1000)
    us = torch.tensor([sorted(per_call)[rounds // 2]], device=device)
    dist.all_reduce(us, op=dist.ReduceOp.MAX)
    return us.item()


def bench(comm, rank, W, device, dtype, tokens, iters):
    import flag_gems

    weight = torch.randn(H, device=device, dtype=dtype)
    if rank == 0:
        print(f"W={W} H={H} max_programs={comm.max_programs} nchannels={comm.nchannels}")
        print("launch (channels, programs):", {T: comm.launch_shape(T) for T in tokens})
        print(f"{'T':>6} {'AR+norm':>10} {'RS+norm+AG':>11} {'fused':>9}  (us per call, median of 5 rounds, max over ranks)")
    for T in tokens:
        x0 = torch.randn(T, H, device=device, dtype=dtype)
        r0 = torch.randn(T, H, device=device, dtype=dtype)
        b, e = row_range(T, W, rank)
        chunk = (T + W - 1) // W
        x, res = x0.clone(), r0.clone()
        xp = torch.zeros(chunk * W, H, device=device, dtype=dtype)
        local = torch.empty(chunk, H, device=device, dtype=dtype)
        full = torch.empty(chunk * W, H, device=device, dtype=dtype)
        rloc = r0[b:e].clone()

        def reset_full():
            x.copy_(x0)
            res.copy_(r0)

        def ar_norm():
            dist.all_reduce(x)
            flag_gems.fused_add_rms_norm(x, res, [H], weight, EPS)

        def reset_rs():
            xp[:T].copy_(x0)
            rloc.copy_(r0[b:e])

        def rs_norm_ag():
            dist.reduce_scatter_tensor(local, xp)
            if e > b:
                flag_gems.fused_add_rms_norm(local[: e - b], rloc, [H], weight, EPS)
            dist.all_gather_into_tensor(full, local)

        def reset_fused():
            x.copy_(x0)
            rloc.copy_(r0[b:e])

        def fused():
            comm(x, rloc, weight, EPS)

        t_ar = _time(ar_norm, reset_full, iters, device)
        t_rs = _time(rs_norm_ag, reset_rs, iters, device)
        t_fu = _time(fused, reset_fused, iters, device)
        if rank == 0:
            print(f"{T:>6} {t_ar:>10.1f} {t_rs:>11.1f} {t_fu:>9.1f}")
    assert comm.errors() == 0, f"{comm.errors()} barrier timeouts"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", action="store_true")
    p.add_argument("--tokens", default="512,1024,2048,4096,8192,16384")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--nchannels", type=int, default=28)
    args = p.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    cpu_group = dist.new_group(backend="gloo")
    rank, W = dist.get_rank(), dist.get_world_size()
    dtype = torch.bfloat16
    tokens = [int(t) for t in args.tokens.split(",")]
    max_tokens = max(tokens + [4096]) if args.bench else 4096

    # rank i runs on physical PPU visible[i]; pccl's 810E rings restricted to them
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    phys = [int(v) for v in visible.split(",")] if visible else list(range(W))
    rings = ppu810e_rings(phys[:W])
    comm = fused_rs_add_rms_norm_ag(
        dist.group.WORLD,
        device,
        max_tokens,
        H,
        dtype,
        cpu_group=cpu_group,
        rings=rings,
        nchannels=args.nchannels,
    )
    qcomm = fused_qkv_norm_ag(
        dist.group.WORLD,
        device,
        max_tokens,
        Q_DIM,
        KV_DIM,
        ROW_DIM,
        dtype,
        cpu_group=cpu_group,
        rings=rings,
        nchannels=args.nchannels,
    )
    if rank == 0:
        print(f"PPUs {phys[:W]}, {len(rings)} rings:", [[phys[i] for i in r] for r in rings])
    try:
        check(comm, rank, W, device, dtype)
        check_rs_only(comm, rank, W, device, dtype)
        check_qkv(qcomm, rank, W, device, dtype)
        compare_baseline(comm, rank, W, device, dtype)
        if args.bench:
            bench(comm, rank, W, device, dtype, tokens, args.iters)
            bench_split(comm, qcomm, rank, W, device, dtype, tokens, args.iters)
    finally:
        qcomm.close()
        comm.close()
        dist.destroy_process_group()


@pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="needs torchrun")
def test_rs_norm_ag():
    main()


if __name__ == "__main__":
    main()
