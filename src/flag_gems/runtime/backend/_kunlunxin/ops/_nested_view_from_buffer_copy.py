import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("flag_gems." + __name__)

# Payload copy block size (in elements). B=8192 is a fixed value validated on 0.4–0.8MB payloads; no sweep was run.
_CPO_BLOCK = 8192

# Upper bound on the component count for padding metadata with the scalar-loop kernel.
# Measured (2026-09-15, same-window A/B, 3-round re-check): NC≤16 is ~23% faster than `copy_`; NC≥24 is slower
# (scalar loop ~1.4µs per iteration, a serial-latency and not a bandwidth effect; 24→1.03× / 28→1.17× / 32→1.24×)
# ⇒ **16** is taken as a conservative threshold; above it batches use the pre-change `copy_` (zero regression).
_PAD_SCALAR_MAX = 16


@triton.jit
def _copy_payload_kernel(self_ptr, values_ptr, NP, S0, BLOCK: tl.constexpr):
    """Payload copy: copies NP elements following `self`'s stride.

    `values` is `empty_strided(self.shape, self.stride())`, so both share the same
    layout ⇒ the same indices are valid for both.
    """
    pid = tl.program_id(0)
    idx = pid * BLOCK + tl.arange(0, BLOCK)
    m = idx < NP
    off = idx * S0
    tl.store(values_ptr + off, tl.load(self_ptr + off, mask=m), mask=m)


@triton.jit
def _pad_offsets_kernel(off_ptr, fo_ptr, NC, SO):
    """Writes `offsets` into `full_offsets[:NC]` and fills the last entry with `full_offsets[NC] = offsets[0]`.

    Note: using a **scalar loop** (no constexpr, no mask) is deliberate: the 2026-09-15
    correctness check found that putting the "small payload mask" and the "masked vector
    store" into **the same kernel** silently drops the middle of the metadata (always
    present for `numel ≤ 256 & NC ≥ 2`; the ablation shows metadata is fine once it moves
    to a scalar loop, the mechanism was not traced in the IR, so for now it is worked
    around by splitting the two). Once the two tasks are split into two launches, this
    kernel's shape matches the "good" variant from the ablation. The scalar loop also
    avoids the per-component-count recompilation that making `META` a constexpr causes.
    """
    for i in range(0, NC):
        tl.store(fo_ptr + i, tl.load(off_ptr + i * SO))
    tl.store(fo_ptr + NC, tl.load(off_ptr))


# On XPU (xpytorch) the customized aten._nested_view_from_buffer / _copy implementations
# assert buffer_storage_size == total number of component elements, and reading back a
# nested tensor built by _nested_view_from_buffer (unbind / index) segfaults outright;
# so the only usable way to construct nested tensors on the Kunlunxin backend is the
# torch.nested family of APIs.
#
# Performance fix (vs the previous version: empty_strided + _copy_from snapshot + as_nested_tensor assembly):
#   1. In the previous version `torch.nested.as_nested_tensor` internally went through
#      `_nested_tensor_from_tensor_list` → `torch.cat`, and `cat` happens to be an
#      operator overridden by FlagGems: under use_gems, 3 unequal-length components hit
#      the generic dim-0 path in cat.py (3 Triton copy launches), which alone costs
#      ~0.2ms; plus 9 `.item()` host synchronizations (~0.13ms), so use_gems steady
#      state is ~0.4ms;
#   2. Switched to building a **jagged layout** view through
#      `_nested_view_from_values_offsets_lengths` (`torch._nested_view_from_jagged`):
#      component lengths (lengths) are passed in explicitly, so arbitrary offsets
#      (including holes / overlaps) map directly to `values[offsets[i]:+len_i]`, matching
#      the reference semantics. The whole path only uses metadata primitives
#      (empty_strided / _nested_view_from_jagged) plus our own Triton copies.
#      2026-09-15: the copy was originally split into 3 `copy_()` calls (payload + two
#      32B metadata blocks), each paying ~24µs of Python dispatch overhead under use_gems
#      (the 3 copy_ dispatches total **≈107µs**, measured in isolation; the whole "3→1
#      fusion" probe saves ~150µs in total, while **the device side is only ~6µs** --
#      90 kernels / 30 calls) ⇒ changed to **two bare kernel launches** (payload and
#      metadata separately; the reason they are not fused into one is in
#      `_pad_offsets_kernel`).
#   3. Restriction: jagged components are contiguous (stride-1) 1-D views, so the fast
#      path only applies when self is 1-D, nested_size is (N,1) int64, all strides are 1,
#      offsets is 1-D int64 with length ≥ the number of components, and `numel * stride`
#      stays within the int32 index range; anything else falls back to the generic
#      `as_nested_tensor` path (which keeps arbitrary stride/dimension semantics).
def _nested_view_from_buffer_copy(
    self: torch.Tensor,
    nested_size: torch.Tensor,
    nested_strides: torch.Tensor,
    offsets: torch.Tensor,
):
    logger.debug("GEMS_KUNLUNXIN _NESTED_VIEW_FROM_BUFFER_COPY")
    num_components = nested_size.shape[0]

    if (
        self.dim() == 1
        and nested_size.dim() == 2
        and nested_size.shape[1] == 1
        and nested_size.dtype == torch.int64
        and nested_strides.dtype == torch.int64
        and offsets.dtype == torch.int64
        and offsets.dim() == 1
        and offsets.numel() >= max(1, num_components)
        and all(s == 1 for s in nested_strides.reshape(-1).tolist())
        and self.numel() * max(1, self.stride(0)) < 2**31
    ):
        # Payload copied in one go (op copy semantics; the nested tensor is a view of
        # `values`), metadata (offsets padded to num_components+1) from the second launch.
        values = torch.empty_strided(
            self.shape, self.stride(), dtype=self.dtype, device=self.device
        )
        full_offsets = torch.empty_strided(
            (num_components + 1,), (1,), dtype=torch.int64, device=self.device
        )
        _copy_payload_kernel[(max(1, triton.cdiv(self.numel(), _CPO_BLOCK)),)](
            self, values, self.numel(), self.stride(0), _CPO_BLOCK
        )
        if num_components <= _PAD_SCALAR_MAX:
            _pad_offsets_kernel[(1,)](
                offsets, full_offsets, num_components, offsets.stride(0)
            )
        else:
            # Many components: the batch uses the pre-change `copy_` (the scalar loop is
            # slow at large NC), but the **last entry still uses the kernel above**. Avoids
            # `full_offsets[n:].copy_(offsets[:1])`: `is_contiguous()` is always true for a
            # 1-element tensor, so the tle fast path of gem copy_ misfires
            # (`TensorDescriptor` asserts the last dim has stride==1) and **always throws
            # on a cold run** -- a pre-existing defect on the copy_ side, which this kernel
            # sidesteps by addressing through `stride(0)`.
            full_offsets[:num_components].copy_(offsets)
            _pad_offsets_kernel[(1,)](
                offsets, full_offsets[num_components:], 0, offsets.stride(0)
            )
        # Reuse the project's own `_nested_view_from_jagged` gem: building the
        # jagged view via torch's `nested_view_from_values_offsets_lengths` (or
        # its internal `NestedTensor` class) makes this path a torch call; the
        # gem below constructs the same object from in-repo code.
        from flag_gems.ops import _nested_view_from_jagged

        return _nested_view_from_jagged(
            values,
            full_offsets,
            values,  # unused dummy kept for ATen signature compatibility
            lengths=nested_size[:, 0],
            ragged_idx=1,
        )

    # Generic fallback: per-component as_strided views of a snapshot copy.
    snapshot = torch.empty_strided(
        self.shape, self.stride(), dtype=self.dtype, device=self.device
    )
    snapshot.copy_(self)

    components = []
    for i in range(num_components):
        size_i = int(nested_size[i].item())
        stride_i = (
            int(nested_strides[i].item())
            if nested_strides.ndim > 1
            else int(nested_strides[i].item())
        )
        offset_i = int(offsets[i].item())
        components.append(snapshot.as_strided((size_i,), (stride_i,), offset_i))

    return torch.nested.as_nested_tensor(components)


__all__ = ["_nested_view_from_buffer_copy"]
