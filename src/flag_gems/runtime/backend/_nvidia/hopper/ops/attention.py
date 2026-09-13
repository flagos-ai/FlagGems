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

"""Single vLLM-compatible Python entry point for Hopper FA2/FA3 attention."""

from flag_gems.ops.attention import flash_attn_varlen_func as _generic_fa2

from .attention_impl.launcher import launch_fa3
from .attention_impl.scheduling import FA3Scheduler
from .attention_impl.validation import prepare_fa3_inputs, validate_fa3_plan


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size: list[int] | None = None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    s_aux=None,
    num_splits: int = 0,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    fa_version: int = 2,
):
    """Run the vLLM varlen contract through one FA2/FA3 Python interface.

    vLLM chooses the FlashAttention version before calling this function.  Once
    version 3 is requested, shape-dependent dispatch stays inside the TLE FA3
    implementation; the default ``auto`` route does not silently retry a
    different backend.  Unsupported FA3 inputs or runtime capabilities fail at
    the preparation/plan boundary.
    """

    # fa_version is the backend generation selected by vLLM at model setup.
    if type(fa_version) is not int or fa_version not in (2, 3):
        raise RuntimeError(f"Unsupported fa_version={fa_version}")

    if fa_version == 2:
        # Call the module implementation, not the public override, to avoid
        # recursively calling this Hopper entry point.
        return _generic_fa2(
            q,
            k,
            v,
            max_seqlen_q,
            cu_seqlens_q,
            max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            q_v=q_v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            softcap=softcap,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
            return_softmax_lse=return_softmax_lse,
            out=out,
            scheduler_metadata=scheduler_metadata,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            s_aux=s_aux,
            num_splits=num_splits,
            cp_world_size=cp_world_size,
            cp_rank=cp_rank,
            cp_tot_seqused_k=cp_tot_seqused_k,
            fa_version=2,
        )

    inputs = prepare_fa3_inputs(
        q=q,
        k=k,
        v=v,
        max_seqlen_q=max_seqlen_q,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=max_seqlen_k,
        cu_seqlens_k=cu_seqlens_k,
        seqused_k=seqused_k,
        q_v=q_v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=window_size,
        softcap=softcap,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        return_softmax_lse=return_softmax_lse,
        block_table=block_table,
        out=out,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        num_splits=num_splits,
        s_aux=s_aux,
        cp_world_size=cp_world_size,
        cp_rank=cp_rank,
        cp_tot_seqused_k=cp_tot_seqused_k,
    )
    plan = FA3Scheduler.build(inputs)
    validate_fa3_plan(inputs, plan)
    if plan.log_plan:
        print(
            "FLAG_GEMS_FA3_TLE_ROUTE "
            f"layout={'paged' if inputs.is_paged else 'dense'} "
            f"mode={plan.workload} family={plan.kernel.value} "
            f"reason={plan.reason}"
        )
    result, softmax_lse = launch_fa3(inputs, plan)
    return (result, softmax_lse) if inputs.return_softmax_lse else result


__all__ = ["flash_attn_varlen_func"]
