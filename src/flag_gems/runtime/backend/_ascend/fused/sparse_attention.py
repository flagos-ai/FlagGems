import logging
import os

import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import HAS_TLE

from ..utils import CORE_NUM

# Enable all blocks parallel to avoid coreDim > 65535 issue on NPU
os.environ["TRITON_ALL_BLOCKS_PARALLEL"] = "1"

if HAS_TLE:
    try:
        import triton.experimental.tle as tle
    except ImportError:
        tle = None
else:
    tle = None

logger = logging.getLogger("flag_gems").getChild(__name__.lstrip("."))


@triton.jit
def fused_pa_rope_to_sparse_kernel(
    k_pa_ptr,
    k_rope_pa_ptr,
    v_pa_ptr,
    block_table_ptr,
    sparse_indices_ptr,
    k_sparse_out_ptr,
    v_sparse_out_ptr,
    stride_k_pa_bn,
    stride_k_pa_bs,
    stride_k_pa_n,
    stride_k_pa_d,
    stride_k_rope_pa_bn,
    stride_k_rope_pa_bs,
    stride_k_rope_pa_n,
    stride_k_rope_pa_d,
    stride_v_pa_bn,
    stride_v_pa_bs,
    stride_v_pa_n,
    stride_v_pa_d,
    stride_bt_b,
    stride_bt_blk,
    stride_si_b,
    stride_si_n,
    stride_si_topk,
    stride_out_b,
    stride_out_n,
    stride_out_topk,
    stride_out_d,
    stride_v_b,
    stride_v_n,
    stride_v_topk,
    stride_v_d,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_DK_ROPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for b_idx in range(B):
        b = b_idx
        for idx in range(pid, TOPK, num_programs):
            n = 0
            sparse_idx = tl.load(
                sparse_indices_ptr
                + b * stride_si_b
                + n * stride_si_n
                + idx * stride_si_topk
            )
            block_id = sparse_idx // BLOCK_SIZE
            bs_offset = sparse_idx % BLOCK_SIZE
            actual_block_id = tl.load(
                block_table_ptr + b * stride_bt_b + block_id * stride_bt_blk
            )
            k_pa_offset = (
                actual_block_id * stride_k_pa_bn
                + bs_offset * stride_k_pa_bs
                + n * stride_k_pa_n
            )
            k_rope_pa_offset = (
                actual_block_id * stride_k_rope_pa_bn
                + bs_offset * stride_k_rope_pa_bs
                + n * stride_k_rope_pa_n
            )
            v_pa_offset = (
                actual_block_id * stride_v_pa_bn
                + bs_offset * stride_v_pa_bs
                + n * stride_v_pa_n
            )
            k_vec = tl.load(
                k_pa_ptr + k_pa_offset + tl.arange(0, BLOCK_DK) * stride_k_pa_d
            )
            v_vec = tl.load(
                v_pa_ptr + v_pa_offset + tl.arange(0, BLOCK_DV) * stride_v_pa_d
            )
            out_offset = b * stride_out_b + n * stride_out_n + idx * stride_out_topk
            out_offset_v = b * stride_v_b + n * stride_v_n + idx * stride_v_topk

            if BLOCK_DK_ROPE > 0:
                full_k = tl.full((BLOCK_DK + BLOCK_DK_ROPE,), 0.0, dtype=tl.float16)
                k_rope_vec = tl.load(
                    k_rope_pa_ptr
                    + k_rope_pa_offset
                    + tl.arange(0, BLOCK_DK_ROPE) * stride_k_rope_pa_d
                )
                full_k = tle.dsa.insert_slice(
                    full_k, k_vec, offsets=(0,), sizes=(BLOCK_DK,), strides=(1,)
                )
                full_k = tle.dsa.insert_slice(
                    full_k,
                    k_rope_vec,
                    offsets=(BLOCK_DK,),
                    sizes=(BLOCK_DK_ROPE,),
                    strides=(1,),
                )
                tl.store(
                    k_sparse_out_ptr
                    + out_offset
                    + tl.arange(0, BLOCK_DK + BLOCK_DK_ROPE) * stride_out_d,
                    full_k,
                )
            else:
                tl.store(
                    k_sparse_out_ptr
                    + out_offset
                    + tl.arange(0, BLOCK_DK) * stride_out_d,
                    k_vec,
                )
            tl.store(
                v_sparse_out_ptr + out_offset_v + tl.arange(0, BLOCK_DV) * stride_v_d,
                v_vec,
            )


def triton_fused_pa_rope_to_sparse(
    k_pa, k_rope_pa, v_pa, block_table, sparse_indices, block_size
):
    block_num, _, n, dk = k_pa.shape
    B = block_table.shape[0]
    TOPK = sparse_indices.size(-1)
    N = 1
    _, _, _, dv = v_pa.shape

    has_rope = k_rope_pa is not None
    dk_rope = k_rope_pa.shape[-1] if has_rope else 0
    dk_total = dk + dk_rope
    k_sparse = torch.empty((B, N, TOPK, dk_total), dtype=k_pa.dtype, device=k_pa.device)
    v_sparse = torch.empty((B, N, TOPK, dv), dtype=v_pa.dtype, device=v_pa.device)
    grid = (min(48, TOPK),)
    sparse_indices_input = sparse_indices
    if sparse_indices.dim() == 2:
        sparse_indices_input = sparse_indices.unsqueeze(1)
    k_rope_pa_input = k_rope_pa if has_rope else k_pa
    fused_pa_rope_to_sparse_kernel[grid](
        k_pa,
        k_rope_pa_input,
        v_pa,
        block_table,
        sparse_indices_input,
        k_sparse,
        v_sparse,
        k_pa.stride(0),
        k_pa.stride(1),
        k_pa.stride(2),
        k_pa.stride(3),
        k_rope_pa_input.stride(0),
        k_rope_pa_input.stride(1),
        k_rope_pa_input.stride(2),
        k_rope_pa_input.stride(3),
        v_pa.stride(0),
        v_pa.stride(1),
        v_pa.stride(2),
        v_pa.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        sparse_indices_input.stride(0),
        sparse_indices_input.stride(1),
        sparse_indices_input.stride(2),
        k_sparse.stride(0),
        k_sparse.stride(1),
        k_sparse.stride(2),
        k_sparse.stride(3),
        v_sparse.stride(0),
        v_sparse.stride(1),
        v_sparse.stride(2),
        v_sparse.stride(3),
        BLOCK_DK=dk,
        BLOCK_DV=dv,
        BLOCK_DK_ROPE=dk_rope,
        TOPK=TOPK,
        BLOCK_SIZE=block_size,
        B=B,
    )

    return k_sparse, v_sparse


@triton.jit
def gather_kv_bnsd_vec_kernel(
    k_ptr,
    v_ptr,
    ind_ptr,
    k_out_ptr,
    v_out_ptr,
    stride_kb,
    stride_kn,
    stride_ks,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vs,
    stride_vd,
    stride_ob,
    stride_on,
    stride_os,
    stride_od,
    stride_ovb,
    stride_ovn,
    stride_ovs,
    stride_ovd,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    TOPK: tl.constexpr,
    B: tl.constexpr,
):
    end = TOPK // 48 * 48
    for b_idx in range(B):
        for batch_start in range(0, end, 48):
            pid_k = tl.program_id(0) + batch_start
            idx = tl.load(ind_ptr + pid_k)
            k_src_off = idx * stride_ks + b_idx * stride_kb
            k_val = tl.load(k_ptr + k_src_off + tl.arange(0, BLOCK_DK) * stride_kd)
            v_src_off = idx * stride_vs + b_idx * stride_vb
            v_val = tl.load(v_ptr + v_src_off + tl.arange(0, BLOCK_DV) * stride_vd)
            k_dst_off = pid_k * stride_os + b_idx * stride_ob
            tl.store(k_out_ptr + k_dst_off + tl.arange(0, BLOCK_DK) * stride_od, k_val)
            v_dst_off = pid_k * stride_ovs + b_idx * stride_ovb
            tl.store(v_out_ptr + v_dst_off + tl.arange(0, BLOCK_DV) * stride_ovd, v_val)
        for batch_start in range(end, TOPK, 48):
            pid_k = tl.program_id(0) + batch_start
            if pid_k < TOPK:
                idx = tl.load(ind_ptr + pid_k)
                k_src_off = idx * stride_ks + b_idx * stride_kb
                k_val = tl.load(k_ptr + k_src_off + tl.arange(0, BLOCK_DK) * stride_kd)
                v_src_off = idx * stride_vs + b_idx * stride_vb
                v_val = tl.load(v_ptr + v_src_off + tl.arange(0, BLOCK_DV) * stride_vd)
                k_dst_off = pid_k * stride_os + b_idx * stride_ob
                tl.store(
                    k_out_ptr + k_dst_off + tl.arange(0, BLOCK_DK) * stride_od, k_val
                )
                v_dst_off = pid_k * stride_ovs + b_idx * stride_ovb
                tl.store(
                    v_out_ptr + v_dst_off + tl.arange(0, BLOCK_DV) * stride_ovd, v_val
                )


def triton_gather_kv_bnsd_vec(k, v, indices):
    B, N, SK, Dk = k.shape
    B, N, SK, Dv = v.shape
    TOPK = indices.size(-1)
    k_sparse = torch.empty((B, N, TOPK, Dk), dtype=k.dtype, device=k.device)
    v_sparse = torch.empty((B, N, TOPK, Dv), dtype=v.dtype, device=v.device)

    grid = (48,)
    gather_kv_bnsd_vec_kernel[grid](
        k,
        v,
        indices.squeeze(0),
        k_sparse,
        v_sparse,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        k_sparse.stride(0),
        k_sparse.stride(1),
        k_sparse.stride(2),
        k_sparse.stride(3),
        v_sparse.stride(0),
        v_sparse.stride(1),
        v_sparse.stride(2),
        v_sparse.stride(3),
        BLOCK_DK=Dk,
        BLOCK_DV=Dv,
        TOPK=TOPK,
        B=B,
    )
    return k_sparse, v_sparse


@triton.jit
def _attn_fwd(
    Q,
    K,
    V,
    O,
    scale_value,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    Q_N: tl.constexpr,
    Q_D: tl.constexpr,
    Q_S: tl.constexpr,
    KV_S: tl.constexpr,
    K_D: tl.constexpr,
    V_D: tl.constexpr,
    sparse_mode: tl.constexpr,
    O_N: tl.constexpr,
    O_D: tl.constexpr,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    blk_size: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    BLOCK_QN_NUM = Q_N // Q_BLOCK_SIZE
    NUM_BLOCKS = B * Q_S * BLOCK_QN_NUM
    pid = tl.program_id(0)
    num_cores = min(CORE_NUM, NUM_BLOCKS)
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        off_b = (block_idx // (Q_S * BLOCK_QN_NUM)).to(tl.int32)
        off_s = ((block_idx // BLOCK_QN_NUM) % Q_S).to(tl.int32)
        off_n = (block_idx % BLOCK_QN_NUM).to(tl.int32)

        q_offset = off_b * stride_qb + off_s * stride_qs
        o_offset = off_b * stride_ob + off_s * stride_os
        k_offset = off_b * stride_kb
        v_offset = off_b * stride_vb

        cur_act_s_q = tl.load(actual_seq_lengths_query + off_b)

        for i in range(cur_act_s_q):
            cur_max = tl.full((Q_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
            logSum = tl.zeros((Q_BLOCK_SIZE,), dtype=tl.float32)
            acc = tl.zeros((Q_BLOCK_SIZE, V_D), dtype=tl.float32)
            q_block_ptr = tl.make_block_ptr(
                base=Q + q_offset,
                shape=(Q_N, Q_D),
                strides=(stride_qn, stride_qd),
                offsets=(off_n * Q_BLOCK_SIZE, 0),
                block_shape=(Q_BLOCK_SIZE, Q_D),
                order=(1, 0),
            )
            q_vec = tl.load(q_block_ptr, boundary_check=(0, 1))
            k_block_ptr = tl.make_block_ptr(
                base=K + k_offset,
                shape=(KV_S, K_D),
                strides=(stride_ks, stride_kd),
                offsets=(0, 0),
                block_shape=(blk_size, K_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=V + v_offset,
                shape=(KV_S, V_D),
                strides=(stride_vs, stride_vd),
                offsets=(0, 0),
                block_shape=(blk_size, V_D),
                order=(1, 0),
            )

            for k_idx in range(KV_S // blk_size):
                k_vec = tl.load(k_block_ptr, boundary_check=(0, 1))
                qk = (
                    tl.dot(q_vec.to(tl.float16), tl.trans(k_vec).to(tl.float16))
                    * scale_value
                )
                block_max = tl.max(qk, axis=1)
                new_max = tl.maximum(cur_max, block_max)
                coeff = tl.math.exp(cur_max - new_max)
                p = tl.math.exp(qk - new_max[:, None])
                logSum = logSum * coeff + tl.sum(p, axis=1)
                v_vec = tl.load(v_block_ptr, boundary_check=(0, 1))
                pv = tl.dot(p.to(tl.float16), v_vec)
                acc = acc * coeff[:, None] + pv
                cur_max = new_max

                k_block_ptr = k_block_ptr.advance((blk_size, 0))
                v_block_ptr = v_block_ptr.advance((blk_size, 0))

            o_block_ptr = tl.make_block_ptr(
                base=O + o_offset,
                shape=(O_N, O_D),
                strides=(stride_on, stride_od),
                offsets=(off_n * Q_BLOCK_SIZE, 0),
                block_shape=(Q_BLOCK_SIZE, O_D),
                order=(1, 0),
            )
            acc = acc / logSum[:, None]
            tl.store(o_block_ptr, acc)


@triton.jit
def _attn_fwd_fused_bsnd_to_tnd(
    Q,
    K,
    V,
    O,
    scale_value,
    stride_qb: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qn: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kn: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vn: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_on: tl.constexpr,
    stride_od: tl.constexpr,
    B: tl.constexpr,
    Q_N: tl.constexpr,
    Q_D: tl.constexpr,
    Q_S: tl.constexpr,
    KV_S: tl.constexpr,
    K_D: tl.constexpr,
    V_D: tl.constexpr,
    sparse_mode: tl.constexpr,
    O_N: tl.constexpr,
    O_D: tl.constexpr,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    blk_size: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
):
    BLOCK_QN_NUM = Q_N // Q_BLOCK_SIZE
    NUM_BLOCKS = B * Q_S * BLOCK_QN_NUM
    pid = tl.program_id(0)
    num_cores = min(CORE_NUM, NUM_BLOCKS)
    for block_idx in range(pid, NUM_BLOCKS, num_cores):
        off_b = (block_idx // (Q_S * BLOCK_QN_NUM)).to(tl.int32)
        off_s = ((block_idx // BLOCK_QN_NUM) % Q_S).to(tl.int32)
        off_n = (block_idx % BLOCK_QN_NUM).to(tl.int32)

        q_offset = off_b * stride_qb + off_s * stride_qs
        o_offset = off_b * stride_ot
        k_offset = off_b * stride_kb
        v_offset = off_b * stride_vb

        cur_act_s_q = tl.load(actual_seq_lengths_query + off_b)

        for i in range(cur_act_s_q):
            cur_max = tl.full((Q_BLOCK_SIZE,), float("-inf"), dtype=tl.float32)
            logSum = tl.zeros((Q_BLOCK_SIZE,), dtype=tl.float32)
            acc = tl.zeros((Q_BLOCK_SIZE, V_D), dtype=tl.float32)
            q_block_ptr = tl.make_block_ptr(
                base=Q + q_offset,
                shape=(Q_N, Q_D),
                strides=(stride_qn, stride_qd),
                offsets=(off_n * Q_BLOCK_SIZE, 0),
                block_shape=(Q_BLOCK_SIZE, Q_D),
                order=(1, 0),
            )
            q_vec = tl.load(q_block_ptr, boundary_check=(0, 1))
            k_block_ptr = tl.make_block_ptr(
                base=K + k_offset,
                shape=(KV_S, K_D),
                strides=(stride_ks, stride_kd),
                offsets=(0, 0),
                block_shape=(blk_size, K_D),
                order=(1, 0),
            )
            v_block_ptr = tl.make_block_ptr(
                base=V + v_offset,
                shape=(KV_S, V_D),
                strides=(stride_vs, stride_vd),
                offsets=(0, 0),
                block_shape=(blk_size, V_D),
                order=(1, 0),
            )

            for k_idx in range(KV_S // blk_size):
                k_vec = tl.load(k_block_ptr, boundary_check=(0, 1))
                qk = (
                    tl.dot(q_vec.to(tl.float16), tl.trans(k_vec).to(tl.float16))
                    * scale_value
                )
                block_max = tl.max(qk, axis=1)
                new_max = tl.maximum(cur_max, block_max)
                coeff = tl.math.exp(cur_max - new_max)
                p = tl.math.exp(qk - new_max[:, None])
                logSum = logSum * coeff + tl.sum(p, axis=1)
                v_vec = tl.load(v_block_ptr, boundary_check=(0, 1))
                pv = tl.dot(p.to(tl.float16), v_vec)
                acc = acc * coeff[:, None] + pv
                cur_max = new_max

                k_block_ptr = k_block_ptr.advance((blk_size, 0))
                v_block_ptr = v_block_ptr.advance((blk_size, 0))

            o_block_ptr = tl.make_block_ptr(
                base=O + o_offset,
                shape=(O_N, O_D),
                strides=(stride_on, stride_od),
                offsets=(off_n * Q_BLOCK_SIZE, 0),
                block_shape=(Q_BLOCK_SIZE, O_D),
                order=(1, 0),
            )
            acc = acc / logSum[:, None]
            tl.store(o_block_ptr, acc)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size=1,
        actual_seq_lengths_query=None,
        actual_seq_lengths_kv=None,
        query_rope=None,
        key_rope=None,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=0,
        block_table=None,
    ):
        sparse_indices_orig = sparse_indices.clone()
        total_len = 0
        if layout_query == "TND":
            actual_seq_lengths_query, total_len = trans_tnd_actseq(
                actual_seq_lengths_query
            )
            query, sparse_indices = trans_tnd_to_bsnd_fused(
                query, query_rope, sparse_indices, query.shape, actual_seq_lengths_query
            )
        else:
            if query_rope is not None:
                query = torch.cat([query, query_rope], dim=-1)
        if layout_kv == "PA_BSND":
            block_size = key.shape[1]
            k_sparse, v_sparse = triton_fused_pa_rope_to_sparse(
                key, key_rope, value, block_table, sparse_indices_orig, block_size
            )
            sparse_indices_bnsd = sparse_indices.permute(0, 2, 1, 3).contiguous()
        else:
            if key_rope is not None:
                key = torch.cat([key, key_rope], dim=-1)
            key_bnsd = key.permute(0, 2, 1, 3).contiguous()
            value_bnsd = value.permute(0, 2, 1, 3).contiguous()
            sparse_indices_bnsd = sparse_indices.permute(0, 2, 1, 3).contiguous()

            k_sparse, v_sparse = triton_gather_kv_bnsd_vec(
                key_bnsd, value_bnsd, sparse_indices_bnsd
            )

        k_sparse = k_sparse.contiguous()
        v_sparse = v_sparse.contiguous()
        enable_check_kv_sparse = 0
        if enable_check_kv_sparse:
            key = pa_to_bsnd(key, block_table, actual_seq_lengths_kv)
            key_rope = pa_to_bsnd(key_rope, block_table, actual_seq_lengths_kv)
            value = pa_to_bsnd(value, block_table, actual_seq_lengths_kv)
            if key_rope is not None:
                key = torch.cat([key, key_rope], dim=-1)
            key_bnsd = key.permute(0, 2, 1, 3).contiguous()
            value_bnsd = value.permute(0, 2, 1, 3).contiguous()
            k_sparse_ref, v_sparse_ref = triton_gather_kv_bnsd_vec(
                key_bnsd, value_bnsd, sparse_indices_bnsd
            )
            print(f"k_sparse={k_sparse}")
            print(f"k_sparse_ref={k_sparse_ref}")
            print(f"v_sparse={v_sparse}")
            print(f"v_sparse_ref={v_sparse_ref}")
            assert torch.allclose(
                k_sparse, k_sparse_ref, rtol=1e-5, atol=1e-5
            ), "K_sparse mismatch!"
            assert torch.allclose(
                v_sparse, v_sparse_ref, rtol=1e-5, atol=1e-5
            ), "V_sparse mismatch!"
        num_cores = CORE_NUM
        out_shape_bsnd = list(query.shape)
        if query_rope is not None:
            out_shape_bsnd[-1] = out_shape_bsnd[-1] - query_rope.shape[-1]
        B, Q_S, Q_N, Q_D = query.shape
        _, _, KV_S, K_D = k_sparse.shape

        if layout_query == "TND":
            output = torch.empty(
                (total_len, out_shape_bsnd[2], out_shape_bsnd[3]),
                device=query.device,
                dtype=torch.float32,
            )
            _attn_fwd_fused_bsnd_to_tnd[(num_cores,)](
                query,
                k_sparse,
                v_sparse,
                output,
                scale_value,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),
                k_sparse.stride(0),
                k_sparse.stride(1),
                k_sparse.stride(2),
                k_sparse.stride(3),
                v_sparse.stride(0),
                v_sparse.stride(1),
                v_sparse.stride(2),
                v_sparse.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                B=B,
                Q_N=Q_N,
                Q_D=Q_D,
                Q_S=Q_S,
                KV_S=KV_S,
                K_D=K_D,
                V_D=v_sparse.shape[3],
                sparse_mode=sparse_mode,
                O_N=output.shape[1],
                O_D=output.shape[2],
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                blk_size=128,
                Q_BLOCK_SIZE=16,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )

        else:
            output = torch.empty(
                out_shape_bsnd, device=query.device, dtype=torch.float32
            )
            _attn_fwd[(num_cores,)](
                query,
                k_sparse,
                v_sparse,
                output,
                scale_value,
                query.stride(0),
                query.stride(1),
                query.stride(2),
                query.stride(3),
                k_sparse.stride(0),
                k_sparse.stride(1),
                k_sparse.stride(2),
                k_sparse.stride(3),
                v_sparse.stride(0),
                v_sparse.stride(1),
                v_sparse.stride(2),
                v_sparse.stride(3),
                output.stride(0),
                output.stride(1),
                output.stride(2),
                output.stride(3),
                B=B,
                Q_N=Q_N,
                Q_D=Q_D,
                Q_S=Q_S,
                KV_S=KV_S,
                K_D=K_D,
                V_D=v_sparse.shape[3],
                sparse_mode=sparse_mode,
                O_N=output.shape[2],
                O_D=output.shape[3],
                actual_seq_lengths_query=actual_seq_lengths_query,
                actual_seq_lengths_kv=actual_seq_lengths_kv,
                blk_size=128,
                Q_BLOCK_SIZE=16,
                limit_auto_multi_buffer_only_for_local_buffer=False,
                limit_auto_multi_buffer_of_local_buffer="no-limit",
            )
            output = output.permute(0, 2, 1, 3).contiguous()

        ctx.save_for_backward(query, k_sparse, v_sparse, output)
        ctx.scale_value = scale_value
        return output


def pa_to_bsnd(pa_in, block_table, actual_seq_lengths):
    block_num, block_size, n, d = pa_in.shape
    b = len(actual_seq_lengths)
    output = torch.empty(
        (b, block_num * block_size // b, 1, d), dtype=pa_in.dtype, device=pa_in.device
    )
    for i in range(b):
        for j in range(20):
            output[i, j * block_size : (j + 1) * block_size, 0, :] = pa_in[
                block_table[i][j], :, 0, :
            ].reshape(block_size, d)
    return output


@triton.jit
def trans_tnd_to_bsnd_fused_kernel(
    query_ptr,
    query_rope_ptr,
    sparse_ptr,
    query_out_ptr,
    sparse_out_ptr,
    act_s,
    stride_q_t,
    stride_q_tn,
    stride_q_td,
    stride_qr_t,
    stride_qr_tn,
    stride_qr_td,
    stride_s_t,
    stride_s_tn,
    stride_s_td,
    stride_qob,
    stride_qobs,
    stride_qon,
    stride_qod,
    stride_sb,
    stride_sbs,
    stride_sbn,
    stride_sbd,
    B: tl.constexpr,
    N: tl.constexpr,
    D_QUERY: tl.constexpr,
    D_ROPE: tl.constexpr,
    D_SPARSE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D_QUERY: tl.constexpr,
    BLOCK_D_ROPE: tl.constexpr,
    BLOCK_D_SPARSE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_head_blocks = (N + BLOCK_N - 1) // BLOCK_N
    t_idx = tl.full((), 0, dtype=tl.int64)
    for tn_id in range(B):
        if pid == 0:
            sparse_block_ptr = tl.make_block_ptr(
                base=sparse_ptr + t_idx * stride_s_t,
                shape=(1, D_SPARSE),
                strides=(stride_s_tn, stride_s_td),
                offsets=(0, 0),
                block_shape=(1, D_SPARSE),
                order=(1, 0),
            )
            sparse = tl.load(sparse_block_ptr)

            sparse_out_block_ptr = tl.make_block_ptr(
                base=sparse_out_ptr + t_idx * stride_sb,
                shape=(1, D_SPARSE),
                strides=(stride_sbn, stride_sbd),
                offsets=(0, 0),
                block_shape=(1, D_SPARSE),
                order=(1, 0),
            )
            tl.store(sparse_out_block_ptr, sparse)
        for head_block_id in range(pid, num_head_blocks, num_programs):
            n_offset = head_block_id * BLOCK_N
            q_block_ptr = tl.make_block_ptr(
                base=query_ptr + t_idx * stride_q_t,
                shape=(N, D_QUERY),
                strides=(stride_q_tn, stride_q_td),
                offsets=(n_offset, 0),
                block_shape=(BLOCK_N, D_QUERY),
                order=(1, 0),
            )
            q_ro_block_ptr = tl.make_block_ptr(
                base=query_rope_ptr + t_idx * stride_qr_t,
                shape=(N, D_ROPE),
                strides=(stride_qr_tn, stride_qr_td),
                offsets=(n_offset, 0),
                block_shape=(BLOCK_N, D_ROPE),
                order=(1, 0),
            )
            q = tl.load(q_block_ptr)
            q_ro = tl.load(q_ro_block_ptr)
            full_q = tl.zeros(
                (BLOCK_N, D_QUERY + D_ROPE), dtype=query_out_ptr.dtype.element_ty
            )
            full_q = tle.dsa.insert_slice(
                full_q, q, offsets=(0, 0), sizes=(BLOCK_N, D_QUERY), strides=(1, 1)
            )
            full_q = tle.dsa.insert_slice(
                full_q,
                q_ro,
                offsets=(0, D_QUERY),
                sizes=(BLOCK_N, D_ROPE),
                strides=(1, 1),
            )

            q_out_block_ptr = tl.make_block_ptr(
                base=query_out_ptr + t_idx * stride_qob,
                shape=(N, D_QUERY + D_ROPE),
                strides=(stride_qon, stride_qod),
                offsets=(n_offset, 0),
                block_shape=(BLOCK_N, D_QUERY + D_ROPE),
                order=(1, 0),
            )
            tl.store(q_out_block_ptr, full_q)
        t_idx = t_idx + tl.load(act_s + tn_id)


def trans_tnd_to_bsnd_fused(
    query, query_rope, sparse_indices, shape, act_seq, grid=(16,)
):
    t, n, d_query = shape
    b = len(act_seq)
    s = max(act_seq)
    d_rope = query_rope.shape[2] if query_rope is not None else 0
    d_sparse = sparse_indices.shape[2]
    d_query_out = d_query + d_rope
    query_out = torch.empty(
        (b, s, n, d_query_out), dtype=query.dtype, device=query.device
    )
    sparse_out = torch.empty(
        (b, s, 1, d_sparse), dtype=sparse_indices.dtype, device=sparse_indices.device
    )
    assert sparse_indices.shape[1] == 1, "sparse_indices second dim must be 1 when MLA"
    block_n = min(16, n)
    num_head_blocks = (n + block_n - 1) // block_n
    num_programs = min(CORE_NUM, num_head_blocks)

    trans_tnd_to_bsnd_fused_kernel[num_programs,](
        query,
        query_rope,
        sparse_indices,
        query_out,
        sparse_out,
        act_seq,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        query_rope.stride(0),
        query_rope.stride(1),
        query_rope.stride(2),
        sparse_indices.stride(0),
        sparse_indices.stride(1),
        sparse_indices.stride(2),
        query_out.stride(0),
        query_out.stride(1),
        query_out.stride(2),
        query_out.stride(3),
        sparse_out.stride(0),
        sparse_out.stride(1),
        sparse_out.stride(2),
        sparse_out.stride(3),
        B=b,
        N=n,
        D_QUERY=d_query,
        D_ROPE=d_rope,
        D_SPARSE=d_sparse,
        BLOCK_N=block_n,
        BLOCK_D_QUERY=d_query,
        BLOCK_D_ROPE=d_rope,
        BLOCK_D_SPARSE=d_sparse,
    )
    return query_out, sparse_out


def trans_tnd_actseq(seq):
    device = seq.device if isinstance(seq, torch.Tensor) else None
    if isinstance(seq, torch.Tensor):
        seq = seq.cpu().tolist()
    list_len = len(seq)
    output = []
    output = [seq[0]]
    total_len = seq[0]
    for i in range(list_len - 1):
        new_item = seq[i + 1] - seq[i]
        if new_item >= 0:
            output.append(new_item)
            total_len += new_item
        else:
            print(
                f"[ERROR]trans_tnd_actseq: Wrong input actseq:{seq}, in loop {i}, item {new_item} < 0"
            )
    return torch.tensor(output, device=device), total_len


def sparse_attention(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    sparse_block_size=1,
    actual_seq_lengths_query=None,
    actual_seq_lengths_kv=None,
    query_rope=None,
    key_rope=None,
    layout_query="BSND",
    layout_kv="BSND",
    sparse_mode=0,
    block_table=None,
):
    return _attention.apply(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size,
        actual_seq_lengths_query,
        actual_seq_lengths_kv,
        query_rope,
        key_rope,
        layout_query,
        layout_kv,
        sparse_mode,
        block_table,
    )


# ---------------------------------------------------------------------------
# Triton kernel: sparse attention with attention-sink
# Adapted for Ascend NPU: 1D grid, tiling for UB overflow
# ---------------------------------------------------------------------------
@triton.jit
def sparse_attn_triton_kernel(
    Q,  # (b, m, h, d)  bf16
    KV,  # (b, n, d)     bf16
    O,  # (b, m, h, d)  bf16
    attn_sink,  # (h,)          fp32
    topk_idxs,  # (b, m, topk)  int32
    stride_qb,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kvb,
    stride_kvn,
    stride_kvd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    stride_idxb,
    stride_idxm,
    stride_idxk,
    scale,
    topk,
    kv_len,
    H_ACTUAL,
    BLOCK: tl.constexpr,
    BLOCK_SUB: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BATCH_STRIDE: tl.constexpr,
):
    # 1D grid: each task handles one (batch, seq_pos)
    pid = tl.program_id(0)
    pid_b = pid // BATCH_STRIDE
    pid_m = pid % BATCH_STRIDE

    # ---- load Q matrix: (H, D) — all heads at once ----
    q_base = Q + pid_b * stride_qb + pid_m * stride_qm
    offs_h = tl.arange(0, H)
    offs_d = tl.arange(0, D)
    h_mask = offs_h < H_ACTUAL
    q_ptrs = q_base + offs_h[:, None] * stride_qh + offs_d[None, :] * stride_qd
    q_block = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)  # (H, D) bf16

    # ---- base pointers ----
    kv_base = KV + pid_b * stride_kvb
    idx_base = topk_idxs + pid_b * stride_idxb + pid_m * stride_idxm

    # ---- online softmax state ----
    acc_o = tl.zeros([H, D], dtype=tl.float32)
    scores_max = tl.full([H], float("-inf"), dtype=tl.float32)
    sum_exp = tl.zeros([H], dtype=tl.float32)

    # Two-level tiling: BLOCK (outer) -> BLOCK_SUB (inner)
    num_block_iter = (topk + BLOCK - 1) // BLOCK
    num_sub_iter = (BLOCK + BLOCK_SUB - 1) // BLOCK_SUB
    offs_blk = tl.arange(0, BLOCK_SUB)

    for t in range(num_block_iter):
        block_start = t * BLOCK
        for s in range(num_sub_iter):
            # Process BLOCK elements in sub-tiles
            sub_start = block_start + s * BLOCK_SUB
            raw_offs = sub_start + offs_blk  # (BLOCK_SUB,)
            idx_mask = raw_offs < topk
            idxs = tl.load(
                idx_base + raw_offs * stride_idxk, mask=idx_mask, other=0
            )  # (BLOCK_SUB,)

            # Clamp negative indices to 0 (matching PyTorch behavior on NPU)
            idxs = tl.where(idxs < 0, 0, idxs)

            # Check index validity: idxs must be >= 0 and < kv_len
            # Create valid mask based on both position and index value
            index_valid = (idxs >= 0) & (idxs < kv_len)
            valid_mask = idx_mask & index_valid  # (BLOCK_SUB,)

            # -- gather KV block: (BLOCK_SUB, D) --
            kv_ptrs = (
                kv_base + idxs[:, None] * stride_kvn + offs_d[None, :] * stride_kvd
            )
            kv_block = tl.load(
                kv_ptrs, mask=valid_mask[:, None], other=0.0
            )  # (BLOCK_SUB, D) bf16

            # -- scores: Q @ KV^T -> (H, BLOCK_SUB) via GEMM --
            acc_s = tl.dot(
                q_block, tl.trans(kv_block)
            )  # (H, D) @ (D, BLOCK_SUB) = (H, BLOCK_SUB)
            acc_s = acc_s * scale
            # mask invalid positions to -inf
            mask_bias = tl.where(valid_mask, 0.0, float("-inf"))  # (BLOCK_SUB,)
            acc_s = acc_s + mask_bias[None, :]  # broadcast: (H, BLOCK_SUB)

            # -- online softmax update --
            scores_max_prev = scores_max
            block_max = tl.max(acc_s, axis=1)  # (H,)
            scores_max = tl.maximum(scores_max, block_max)

            correction = tl.exp(scores_max_prev - scores_max)  # (H,)
            p = tl.exp(acc_s - scores_max[:, None])  # (H, BLOCK_SUB)

            # -- accumulate output: acc_o = acc_o * correction + P @ KV --
            acc_o = acc_o * correction[:, None]
            acc_o += tl.dot(
                p.to(tl.bfloat16), kv_block
            )  # (H, BLOCK_SUB) @ (BLOCK_SUB, D) = (H, D)
            scores_sum = tl.sum(p, axis=1)  # (H,)
            sum_exp = sum_exp * correction + scores_sum

    # ---- incorporate attn_sink ----
    sink_vals = tl.load(attn_sink + offs_h, mask=h_mask, other=0.0)  # (H,)
    sum_exp = sum_exp + tl.exp(sink_vals - scores_max)

    # ---- normalize ----
    acc_o = acc_o / sum_exp[:, None]

    # ---- store output: (H, D) ----
    o_base = O + pid_b * stride_ob + pid_m * stride_om
    o_ptrs = o_base + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc_o.to(tl.bfloat16), mask=h_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
def _sparse_attn_triton_fallback(q, kv, attn_sink, topk_idxs, softmax_scale):
    b, m, h, d = q.shape
    topk = topk_idxs.shape[-1]
    o = torch.empty_like(q)

    # NPU optimization: use tiling to avoid UB overflow
    # BLOCK: number of KV elements per outer loop iteration
    # BLOCK_SUB: tile size for UB management
    # UB (192KB) constraint: need to fit q_block + kv_block + acc_o + intermediate buffers
    # Use fixed BLOCK to avoid edge cases with non-power-of-2 topk
    BLOCK = 64
    BLOCK_SUB = 16  # smaller chunks to fit UB (192KB), with multi-buffer overhead

    # H must be >= 16 for tl.dot; pad to next power of 2
    H_padded = max(16, triton.next_power_of_2(h))

    # NPU: use 1D grid, TRITON_ALL_BLOCKS_PARALLEL handles large grid
    grid = (b * m,)

    sparse_attn_triton_kernel[grid](
        q,
        kv,
        o,
        attn_sink,
        topk_idxs,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        kv.stride(0),
        kv.stride(1),
        kv.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        topk_idxs.stride(0),
        topk_idxs.stride(1),
        topk_idxs.stride(2),
        softmax_scale,
        topk,
        kv.shape[1],
        h,
        BLOCK=BLOCK,
        BLOCK_SUB=BLOCK_SUB,
        D=d,
        H=H_padded,
        BATCH_STRIDE=m,  # for 1D grid: pid = pid_b * m + pid_m
        num_warps=4,  # reduced for NPU
    )
    return o


def _sparse_attn_tle_adapter(q, kv, attn_sink, topk_idxs, softmax_scale):
    if tle is None:
        raise RuntimeError("TLE support is unavailable in the current Triton build")
    if q.dim() != 4 or kv.dim() != 3 or topk_idxs.dim() != 3:
        raise ValueError("Expected q[B, M, H, D], kv[B, KV, D], topk_idxs[B, M, TOPK]")
    if q.shape[0] != 1 or q.shape[1] != 1:
        raise ValueError(
            "The restored TLE sparse attention path currently only applies to the original B==1, S==1 logic"
        )
    if torch.count_nonzero(attn_sink).item() != 0:
        raise ValueError(
            "The restored TLE sparse attention path follows the original source logic and requires zero attn_sink"
        )

    key = kv[:, :, None, :].contiguous()
    value = key
    sparse_indices = topk_idxs[:, :, None, :].contiguous()
    actual_seq_lengths_query = torch.full(
        (q.shape[0],), q.shape[1], dtype=torch.int32, device=q.device
    )
    actual_seq_lengths_kv = torch.full(
        (q.shape[0],), kv.shape[1], dtype=torch.int32, device=q.device
    )
    output = sparse_attention(
        query=q,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=softmax_scale,
        actual_seq_lengths_query=actual_seq_lengths_query,
        actual_seq_lengths_kv=actual_seq_lengths_kv,
        layout_query="BSND",
        layout_kv="BSND",
    )
    return output.permute(0, 2, 1, 3).contiguous().to(q.dtype)


def sparse_attn_triton(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    if HAS_TLE and tle is not None:
        try:
            return _sparse_attn_tle_adapter(q, kv, attn_sink, topk_idxs, softmax_scale)
        except Exception as exc:
            logger.debug("Falling back to non-TLE sparse attention path: %s", exc)

    return _sparse_attn_triton_fallback(q, kv, attn_sink, topk_idxs, softmax_scale)
