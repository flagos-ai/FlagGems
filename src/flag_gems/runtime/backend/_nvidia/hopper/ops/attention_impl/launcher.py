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

"""ABI lowering, workspace allocation, and kernel dispatch for Hopper FA3."""

import logging

import torch

from flag_gems.runtime import torch_device_fn

from .direct import launch_direct
from .persistent import _reset_scheduler_counter, launch_persistent
from .scheduling import FA3ExecutionPlan, FA3Scheduler, KernelFamily
from .split_combine import combine_persistent_split_kv
from .validation import PreparedFA3Inputs

logger = logging.getLogger(__name__)

# Split-KV's dynamic tile scheduler needs one graph-stable counter.  Keying the
# tiny workspace by a caller-stable tensor address (out when supplied, otherwise
# q) keeps it private to TLE, reusable across eager calls and CUDA Graph capture,
# and independent of any vLLM metadata tensor.
_split_scheduler_counters: dict[tuple, torch.Tensor] = {}


def _get_split_scheduler_counter(owner: torch.Tensor, plan: FA3ExecutionPlan):
    key = (
        owner.device.type,
        owner.device.index,
        owner.data_ptr(),
        plan.persistent_num_splits,
    )
    counter = _split_scheduler_counters.get(key)
    if counter is None:
        counter = torch.empty((1,), dtype=torch.int32, device=owner.device)
        # Split-KV intentionally keeps a monotonic ticket between calls and
        # reduces it modulo the compact work count. Initialize the graph-stable
        # workspace once with a stream-ordered Triton kernel, without adding a
        # reset launch to every invocation.
        _reset_scheduler_counter(counter)
        _split_scheduler_counters[key] = counter
    return counter


def _log_dispatch(
    kernel_name: str,
    *,
    layout: str,
    plan: FA3ExecutionPlan,
) -> None:
    logger.debug(
        "kernel: flash_varlen_fwd_v3_tle kernel=%s layout=%s mode=%s",
        kernel_name,
        layout,
        plan.metadata_mode.value,
    )
    if plan.log_plan:
        if layout == "dense":
            paged_kv_load = "none"
        else:
            paged_kv_load = "non_tma" if plan.paged_kv_non_tma else "tma"
        print(
            "FLAG_GEMS_FA3_TLE_PLAN "
            f"layout={layout} mode={plan.metadata_mode.value} "
            f"kernel={kernel_name} paged_gather="
            f"{FA3Scheduler.paged_gather_name(plan.paged_gather_mode)} "
            "paged_kv_load="
            f"{paged_kv_load} pack_gqa={plan.pack_gqa}"
        )


def launch_fa3(
    inputs: PreparedFA3Inputs,
    plan: FA3ExecutionPlan,
):
    """Materialize and execute a concrete FA3 plan.

    Public and plan validation, runtime initialization, and all path selection
    have already completed. This function only materializes the selected ABI.
    """

    q = inputs.q
    k = inputs.k
    v = inputs.v
    out = inputs.out
    cu_seqlens_q = inputs.cu_seqlens_q
    cu_seqlens_k = inputs.cu_seqlens_k
    seqused_k = inputs.seqused_k
    page_table = inputs.page_table
    alibi_slopes = inputs.alibi_slopes
    is_alibi = alibi_slopes is not None
    alibi_slopes_batch_stride = (
        alibi_slopes.stride(0)
        if alibi_slopes is not None and alibi_slopes.ndim == 2
        else 0
    )
    s_aux = inputs.s_aux
    is_s_aux = s_aux is not None
    q_device = q.device

    max_seqlen_q = inputs.max_seqlen_q
    max_seqlen_k = inputs.max_seqlen_k
    window = inputs.window
    is_causal = window.causal
    is_local = window.local
    window_size_left = window.left
    window_size_right = window.right
    is_paged = inputs.is_paged
    batch_size = inputs.batch_size
    num_heads = inputs.num_heads
    num_heads_k = inputs.num_heads_k
    head_size = inputs.head_dim
    block_size = inputs.block_size
    num_pages = inputs.num_pages
    k_batch_size = num_pages
    page_table_batch_stride = page_table.stride(0)

    k_batch_stride = 0
    total_q = q.size(0)
    pack_factor = plan.pack_factor

    # A cached plan only owns discrete algorithm choices.  Exact launch geometry
    # belongs to the current request and must not be reused from the shape that
    # originally populated the route cache.
    effective_max_q = max_seqlen_q * pack_factor
    effective_num_heads = num_heads_k if plan.pack_gqa else num_heads

    if is_paged:
        # Reuse the otherwise-unused K batch-stride lane for the physical page
        # stride, supporting both NHD and HND cache allocations.
        k_batch_stride = k.stride(0)

    with torch_device_fn.device(q_device):
        if out is not None:
            public_out = out
        else:
            public_out = None
            out = torch.empty_like(q)

        if inputs.return_softmax_lse:
            lse = torch.empty(
                (num_heads, total_q), dtype=torch.float32, device=q_device
            )
            lse_ptr = lse
        else:
            lse = None
            # Every kernel reads this pointer behind STORE_LSE.  A valid existing
            # tensor avoids allocating an output that the public API will discard.
            lse_ptr = q

        # Keep one compact FA3 ABI shared by the direct and persistent kernels.
        # K/V matching strides are validated during input preparation, so the
        # paged V descriptor can reuse k_batch_stride without a duplicate ABI
        # lane. Dropout/softmax-output placeholders from the generic FA2 ABI are
        # intentionally absent because FA3 rejects those features before launch.
        args = (
            q,
            k,
            v,
            out,
            lse_ptr,
            q.stride(-3),
            k.stride(-3),
            v.stride(-3),
            q.stride(-2),
            k.stride(-2),
            v.stride(-2),
            out.stride(-3),
            out.stride(-2),
            k_batch_stride,
            cu_seqlens_q,
            seqused_k is not None,
            cu_seqlens_k,
            seqused_k,
            batch_size,
            k_batch_size,
            num_heads,
            num_heads_k,
            num_heads // num_heads_k,
            max_seqlen_q,
            max_seqlen_k,
            head_size,
            inputs.is_softcap,
            inputs.adjusted_softcap,
            inputs.adjusted_scale_softmax,
            inputs.adjusted_scale_softmax_log2e,
            is_causal,
            is_local,
            window_size_left,
            window_size_right,
            is_paged,
            is_alibi,
            alibi_slopes,
            alibi_slopes_batch_stride,
            is_s_aux,
            s_aux,
            total_q,
            page_table,
            page_table_batch_stride,
            block_size,
        )

        layout = "paged" if is_paged else "dense"
        _log_dispatch(plan.kernel_name, layout=layout, plan=plan)

        if plan.persistent_split_kv:
            scheduler_counter = _get_split_scheduler_counter(
                public_out if public_out is not None else inputs.q,
                plan,
            )
            partial_out = torch.empty(
                (plan.persistent_num_splits, num_heads, total_q, head_size),
                dtype=torch.float32,
                device=q_device,
            )
            partial_lse = torch.empty(
                (plan.persistent_num_splits, num_heads, total_q),
                dtype=torch.float32,
                device=q_device,
            )
            launch_persistent(
                args,
                output=out,
                total_q=total_q,
                head_size=head_size,
                num_sms=inputs.num_sms,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                effective_max_q=effective_max_q,
                effective_num_heads=effective_num_heads,
                pack_factor=pack_factor,
                pack_gqa=plan.pack_gqa,
                paged_gather_mode=plan.paged_gather_mode,
                paged_kv_non_tma=plan.paged_kv_non_tma,
                ragged_scheduler=True,
                heads_in_l2=plan.heads_in_l2,
                dynamic_scheduler=plan.dynamic_scheduler,
                store_lse=inputs.return_softmax_lse,
                partial_out=partial_out,
                partial_lse=partial_lse,
                max_splits=plan.persistent_num_splits,
                scheduler_counter=scheduler_counter,
            )
            combine_persistent_split_kv(
                out,
                lse_ptr,
                partial_out,
                partial_lse,
                scheduler_counter,
                seqused_k,
                cu_seqlens_q,
                batch_size=batch_size,
                num_heads=num_heads,
                max_seqlen_q=max_seqlen_q,
                head_dim=head_size,
                total_q=total_q,
                max_splits=plan.persistent_num_splits,
                store_lse=inputs.return_softmax_lse,
            )
        elif plan.kernel is KernelFamily.DIRECT:
            launch_direct(
                args,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                effective_max_q=effective_max_q,
                effective_num_heads=effective_num_heads,
                pack_factor=pack_factor,
                is_paged=is_paged,
                paged_prefill=plan.paged_prefill_candidate,
                pack_gqa=plan.pack_gqa,
                paged_gather_mode=plan.paged_gather_mode,
                paged_kv_non_tma=plan.paged_kv_non_tma,
                store_lse=inputs.return_softmax_lse,
                total_q=total_q,
                ragged_scheduler=plan.ragged_scheduler,
                heads_in_l2=plan.heads_in_l2,
            )
        elif plan.kernel is KernelFamily.LONG:
            launch_persistent(
                args,
                output=out,
                total_q=total_q,
                head_size=head_size,
                num_sms=inputs.num_sms,
                max_seqlen_k=max_seqlen_k,
                batch_size=batch_size,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                effective_max_q=effective_max_q,
                effective_num_heads=effective_num_heads,
                pack_factor=pack_factor,
                pack_gqa=plan.pack_gqa,
                paged_gather_mode=plan.paged_gather_mode,
                paged_kv_non_tma=plan.paged_kv_non_tma,
                ragged_scheduler=plan.ragged_scheduler,
                heads_in_l2=plan.heads_in_l2,
                dynamic_scheduler=plan.dynamic_scheduler,
                store_lse=inputs.return_softmax_lse,
            )

    return out, lse


__all__ = ["launch_fa3"]
