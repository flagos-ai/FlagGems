# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Select a whole-packet kernel against the ordinary, fully tuned MM family."""

import logging
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
import statistics
import threading

import torch
import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()

# Full-packet geometry, shared by the normal and column output layouts.
_LARGE_PACKET_GEOMETRY = (
    256, 256, 16, 4, 2,
    (1, 2, 4, 8, 16, 32, 512, 1024, 2048, 64, 128, 256),
    (1, 2, 4, 8, 16, 32, 64, 128, 512, 256, 1024, 2048),
)

# Structured buffer descriptors have a 14-bit byte-stride field; BF16 uses 2B.
_STRUCTURED_STRIDE_LIMIT = 1 << 13


def _packet_supported(*, storage_layout=False):
    # Import lazily: released Triton builds need not contain the packet API.
    try:
        from triton.backends.hcu.compiler import HIPOptions
        from triton.experimental.gluon.language.hcu import PacketPrefixLayout
        from triton.flagtune.runtime.graph_benchmark import do_bench_hcu_graph
        if storage_layout:
            from triton.experimental.gluon.language.hcu import PacketStorageLayout
            if PacketStorageLayout is None:
                return False
    except ImportError:
        return False
    return (
        "packet_matrix_read_partition" in HIPOptions.__dataclass_fields__
        and PacketPrefixLayout is not None
        and callable(do_bench_hcu_graph)
    )


def _packet_eligible(a, b, c):
    """Check the byte-address, structured-copy and output ownership contracts."""
    if any(t.ndim != 2 for t in (a, b, c)):
        return False
    m, k = a.shape
    n = b.shape[1]
    nt = b.stride() == (1, k)
    if not _packet_supported(storage_layout=nt):
        return False
    if not (
        m > 1 and n > 1 and k >= 256 and k % 256 == 0
        and (nt or (n % 16 == 0 and k < _STRUCTURED_STRIDE_LIMIT))
        and b.shape[0] == k and c.shape == (m, n)
        and a.dtype == b.dtype == c.dtype == torch.bfloat16
        and a.device == b.device == c.device and a.device.type == "cuda"
        and a.stride() == (k, 1)
        and (nt or b.stride() == (n, 1))
        and c.stride() == (n, 1)
        and not c.requires_grad
        and (not c.is_inference() or torch.is_inference_mode_enabled())
        and c.untyped_storage().data_ptr()
        not in (a.untyped_storage().data_ptr(), b.untyped_storage().data_ptr())
        and all(t.data_ptr() % 16 == 0 for t in (a, b, c))
        and all(2 * x * y < 2**31 for x, y in ((m, k), (k, n), (m, n)))
    ):
        return False
    with torch_device_fn.device(a.device):
        target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.arch == "gfx936"


def _large_packet_variants():
    """Add only modes supported by this compiler/API; no model-shape selector."""
    try:
        from triton._C.libtriton import llvm
        from triton.backends.hcu.compiler import HIPOptions
        from triton.experimental.gluon.language.hcu import WholePacketLayout
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    common = ("mma_phase_schedule", "packet_wait_rotation",
              "lds_dma_alias_workaround", "packet_matrix_read_partition")
    if (WholePacketLayout is None or any(name not in fields for name in common)
            or not callable(getattr(llvm, "prepare_hcu_mma_phase_loops", None))):
        return ()
    modes = []
    if ("packet_post_mma_wait" in fields
            and callable(getattr(llvm, "prepare_hcu_packet_post_mma_wait", None))):
        modes.append("large_safe")
    if ("packet_operand_overlap_diagnostic" in fields
            and callable(getattr(llvm, "prepare_hcu_packet_operand_overlap_diagnostic", None))):
        # Hardware partial-LGKM ordering remains an explicitly unproved
        # boundary. Exact large-geometry per-dispatch pressure has passed;
        # this does not establish a formal hardware ordering guarantee.
        modes.append("large_overlap")
        # Both traversal orders remain measured candidates: group-one improves
        # tall matrices, while the original order can win for wider matrices.
        modes.append("large_overlap_group1")
    return tuple(modes)


def _raw_nt_packet_variants():
    """Require the complete typed API and measured read scheduling options."""
    try:
        from triton._C.libtriton import hcu
        from triton.backends.hcu.compiler import HIPOptions
        from triton.experimental.gluon.language.hcu import structured_register_load
        from triton.experimental.gluon.language.hcu.raw_window import raw_window_owner
    except ImportError:
        return ()
    options = ("structured_register_prepack_address",
               "structured_register_unclamped_column", "priority_after_first_mmac",
               "buffer_store_scalar_offset", "interleave_global_lds_reads",
               "interleave_global_lds_reads_fences")
    bindings = ("create_structured_register_load", "create_raw_window_copy",
                "create_raw_window_read_full", "create_raw_warp_group_id",
                "interleave_global_lds_reads")
    if (any(name not in HIPOptions.__dataclass_fields__ for name in options)
            or any(not callable(getattr(hcu, name, None)) for name in bindings)
            or not callable(structured_register_load) or not callable(raw_window_owner)):
        return ()
    modes = ["raw_nt"]
    if ("tupled_register_fences" in HIPOptions.__dataclass_fields__
            and callable(getattr(hcu, "tuple_register_fences", None))):
        modes.append("raw_nt_wgm4_c4")
    return tuple(modes)


def _launch_raw_nt_packet(a, b, c, *, variant="raw_nt"):
    from .mm_packet_raw_nt_kernel import _mm_packet_raw_nt_kernel
    m, k = a.shape
    n = b.shape[1]
    if b.stride() != (1, k):
        raise ValueError("Raw NT packet mode requires contiguous transposed B storage")
    if variant not in ("raw_nt", "raw_nt_wgm4_c4"):
        raise ValueError("Unsupported raw NT packet mode")
    # GROUP_M is a source constexpr and therefore also separates LibEntry keys.
    # Keep the original traversal available when the measured combination loses.
    traversal = ({"GROUP_M": 4, "tupled_register_fences": True}
                 if variant == "raw_nt_wgm4_c4" else {})
    return _mm_packet_raw_nt_kernel[(triton.cdiv(m, 256), triton.cdiv(n, 256), 1)](
        a, b, c, m, n, k,
        PRIO=True, FENCE=True, GLOBAL_WAIT=False,
        WAVE_SPECIALIZE=True, LGKM_CLEANUP=True,
        C_REBASE=True, FULL_C_TILE=False, STATIC_C_BOUNDS=False,
        num_warps=8, num_stages=1, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", schedule_hint="local-prefetch",
        optimize_epilogue=True,
        lds_dma_alias_workaround=False,
        disable_llvm_loop_unroll=True, mmac_transposed=True, mmac_warps_m=4,
        structured_register_prepack_address=True,
        structured_register_unclamped_column=True,
        priority_after_first_mmac=True, buffer_store_scalar_offset=True,
        interleave_global_lds_reads=True, interleave_global_lds_reads_fences=True,
        buffer_store_split_immediate=False, buffer_store_preserve_masked_base=False,
        rebase_buffer_store=False, fragment_dot_k=False,
        packet_operand_overlap_diagnostic=False, packet_post_mma_wait=False,
        packet_wait_rotation=False, mma_phase_schedule=False,
        packet_reuse_wait="none", packet_redundant_lgkm_cleanup=False,
        packet_matrix_read_partition=False,
        **traversal,
    )


def _loop_carried_refill_supported():
    try:
        from triton._C.libtriton import llvm
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return False
    return (
        "loop_carried_store_refill" in HIPOptions.__dataclass_fields__
        and callable(getattr(llvm, "interleave_hcu_loop_carried_store_refill", None))
    )


def _multi_row_nt_packet_variants():
    # A compile-time tuple of five power-of-two row subtiles shares B.
    # This is a fixed geometry candidate, selected by the ordinary measurement.
    return ("splitu_nt_multi_row_k256",) if _loop_carried_refill_supported() else ()


def _launch_multi_row_nt_k256_packet(a, b, c):
    from .mm_local_splitk_kernel import _mm_local_splitk_kernel

    m, k = a.shape
    n = b.shape[1]
    if (k < 256 or k % 256 or a.stride() != (k, 1)
            or b.stride() != (1, k) or c.stride() != (n, 1)):
        raise ValueError("Multi-row K256 requires contiguous A/C and transposed B")
    return _mm_local_splitk_kernel[(triton.cdiv(m, 80), triton.cdiv(n, 32), 1)](
        a, b.t(), c, m, n, k, 16, 32, M_TILES=5, LOOP_SCHEDULE=3,
        padded_slot_intervals=True, disable_llvm_loop_unroll=True,
        num_warps=4, num_stages=1, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", wave_private_sync=True,
        loop_carried_store_refill=True,
        mma_phase_schedule=False,
        packet_operand_overlap_diagnostic=False, packet_post_mma_wait=False,
        packet_wait_rotation=False, kpack=2, mmac_transposed=False,
    )


def _splitu_nt_packet_variants():
    """Require exact padded-plane dependency analysis for the split-U family."""
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    if "padded_slot_intervals" not in fields:
        return ()
    variants = ("splitu_nt",)
    if all(name in fields for name in ("wave_private_sync", "disable_llvm_loop_unroll")):
        variants += (
            "splitu_nt_k256", "splitu_nt_k256_peeled",
            "splitu_nt_k256_16", "splitu_nt_k256_peeled_16",
        )
        if _loop_carried_refill_supported():
            variants += ("splitu_nt_k256_rotated", "splitu_nt_k256_rotated_16")
    return variants


def _launch_splitu_nt_packet(a, b, c):
    from .mm_local_splitk_kernel import _mm_local_splitk_kernel
    m, k = a.shape
    n = b.shape[1]
    if b.stride() != (1, k):
        raise ValueError("Split-U NT requires contiguous transposed B storage")
    return _mm_local_splitk_kernel[(triton.cdiv(m, 16), triton.cdiv(n, 32), 1)](
        a, b, c, m, n, k, 16, 32,
        INPUT_LAYOUT=2, PADDED=True, ROW_WARP_STRIDE=1, DOUBLE_BUFFER=True,
        padded_slot_intervals=True,
        num_warps=4, num_stages=1, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", wave_private_sync=False,
        mma_phase_schedule=False,
        packet_operand_overlap_diagnostic=False, packet_post_mma_wait=False,
        packet_wait_rotation=False, kpack=2, mmac_transposed=False,
    )


def _launch_splitu_nt_k256_packet(a, b, c, *, peel_tail=False, tile=32,
                                     rotate_inputs=False):
    """Four K64 waves share one tile with wave-private padded input storage."""
    from .mm_local_splitk_kernel import _mm_local_splitk_kernel

    schedule = {}
    if rotate_inputs:
        if not _loop_carried_refill_supported():
            raise ValueError("Rotated inputs require loop-carried refill support")
        schedule["loop_carried_store_refill"] = True
    kernel = _mm_local_splitk_kernel

    m, k = a.shape
    n = b.shape[1]
    if (k < 256 or k % 256 or a.stride() != (k, 1)
            or b.stride() != (1, k) or c.stride() != (n, 1)):
        raise ValueError("K256 NT requires contiguous A/C and transposed contiguous B")
    if tile not in (16, 32):
        raise ValueError("Unsupported K256 NT output tile")
    return kernel[(triton.cdiv(m, tile), triton.cdiv(n, tile), 1)](
        a, b.t(), c, m, n, k, tile, tile,
        LOOP_SCHEDULE=2 if rotate_inputs else 1 if peel_tail else 0,
        padded_slot_intervals=True, disable_llvm_loop_unroll=True,
        num_warps=4, num_stages=1, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", wave_private_sync=True,
        mma_phase_schedule=False,
        packet_operand_overlap_diagnostic=False, packet_post_mma_wait=False,
        packet_wait_rotation=False, kpack=2, mmac_transposed=False,
        **schedule,
    )


# Additional fixed geometries participate in the same measured NN family choice.
# Their producer charts and options match the isolated dotwidth candidates.
_NN_GEOMETRY_OPTIONS = dict(
    num_warps=4, num_stages=5, waves_per_eu=1,
    loop_unroll_stage='post-pipeline', schedule_hint='local-prefetch',
    lds_dma_alias_workaround=True,
    mmac_transposed=True, mmac_warps_m=4,
    packet_matrix_read_partition=True,
    mma_phase_schedule=False, packet_wait_rotation=False,
    packet_post_mma_wait=False, packet_operand_overlap_diagnostic=False,
    packet_reuse_wait='none', packet_redundant_lgkm_cleanup=False,
    rebase_buffer_store=False,
)
_NN_GEOMETRIES = {
    'wide': dict(tile=(128,256,16),warps=(2,2),tiles_per_warp=(2,2),
        A_SOURCE_BASIS=(1,2,4,8,16,32,256,512,1024,64,128,0),
        B_SOURCE_BASIS=(1,2,4,8,16,32,64,128,512,256,1024,2048)),
    'cube': dict(tile=(64,64,64),warps=(2,2),tiles_per_warp=(1,1),
        A_SOURCE_BASIS=(1,2,4,8,16,32,256,512,1024,64,128,2048),
        B_SOURCE_BASIS=(1,2,4,8,16,32,128,512,1024,64,256,2048)),
}
# Keep the same physical packet chart while placing the 2x2 register tiles
# before warp bits in both operands and the accumulator/output layout.
_NN_GEOMETRIES['cube_tpw2'] = dict(
    _NN_GEOMETRIES['cube'], tiles_per_warp=(2, 2))


def _post_mma_packet_variants(*, nt):
    """Offer the audited wait arrangement as another measured family."""
    try:
        from triton._C.libtriton import llvm
        from triton.backends.hcu.compiler import HIPOptions
        from triton.backends.hcu.mma_phase import (
            NATIVE_READINESS_CAPABILITIES,
            check_native_readiness,
        )
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    options = ("mma_phase_schedule", "packet_post_mma_wait",
               "packet_wait_rotation", "packet_redundant_lgkm_cleanup")
    if (any(name not in fields for name in options)
            or "gfx936.v_xad_u32.single_dst_3src" not in NATIVE_READINESS_CAPABILITIES
            or not callable(check_native_readiness)
            or not callable(getattr(llvm, "prepare_hcu_mma_phase_loops", None))
            or not callable(getattr(llvm, "prepare_hcu_packet_post_mma_wait", None))):
        return ()
    geometries = ("rect", "small", "square") if nt else (
        "rect", "small", "square", "wide", "cube", "cube_tpw2")
    return tuple(name + "_post_mma" for name in geometries)


# New measured code-generation modes follow the complete original family list.
# Compiler kwargs do not enter LibEntry's cache key: each options combination
# has its own source constexpr ENTRY_FAMILY (and traversal has GROUP_M).
_CODEGEN_SQUARE_LABEL = "square_tilebase_scalar_preserve"
_CODEGEN_LARGE_MODES = {
    "large_overlap_wait": (8, 2, False),
    "large_overlap_group1_wait": (1, 2, False),
    "large_overlap_wait_fold": (8, 3, True),
    "large_overlap_group1_wait_fold": (1, 3, True),
}
_CODEGEN_SQUARE_OPTIONS = dict(
    _NN_GEOMETRY_OPTIONS,
    compact_packet_owner=True,
    buffer_store_scalar_offset=True,
    buffer_store_split_immediate=True,
    buffer_store_preserve_masked_base=True,
    coalesce_lgkm_waits=False,
    combine_shifted_disjoint_or=False,
)
_CODEGEN_LARGE_OPTIONS = dict(
    num_warps=8, num_stages=5, waves_per_eu=1,
    loop_unroll_stage="post-pipeline", schedule_hint="local-prefetch",
    lds_dma_alias_workaround=True,
    mmac_transposed=True, mmac_warps_m=4,
    packet_matrix_read_partition=True,
    mma_phase_schedule=True, packet_wait_rotation=False,
    packet_post_mma_wait=False, packet_operand_overlap_diagnostic=True,
    packet_reuse_wait="none", packet_redundant_lgkm_cleanup=False,
    rebase_buffer_store=False, coalesce_lgkm_waits=True,
)


def _codegen_packet_variants(*, nt):
    """Append supported code-generation modes without changing the old pool."""
    if nt:
        return ()
    try:
        from triton._C.libtriton import llvm
        from triton.backends.hcu.compiler import HIPOptions
        from triton.backends.hcu.waitcnt_coalesce import coalesce_lgkm_waits
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    if (not callable(coalesce_lgkm_waits)
            or "coalesce_lgkm_waits" not in fields
            or "combine_shifted_disjoint_or" not in fields):
        return ()
    modes = []
    if all(name in fields for name in _CODEGEN_SQUARE_OPTIONS):
        modes.append(_CODEGEN_SQUARE_LABEL)
    if (all(name in fields for name in _CODEGEN_LARGE_OPTIONS)
            and "large_overlap" in _large_packet_variants()
            and "large_overlap_group1" in _large_packet_variants()):
        fold_supported = callable(getattr(llvm, "combine_hcu_shifted_disjoint_or", None))
        modes.extend(name for name, (_, _, fold) in _CODEGEN_LARGE_MODES.items()
                     if not fold or fold_supported)
    return tuple(modes)


_COLUMN_PACKET_MODES = {
    "large_column_chart": 8,
    "large_column_chart_group1": 1,
}
# ENTRY_FAMILY is part of LibEntry's key; compiler-option kwargs are not.
_COLUMN_VECTOR_OFFSET_MODES = {
    "large_column_chart_vector_offsets": 8,
    "large_column_chart_group1_vector_offsets": 1,
}


def _column_packet_variants(*, nt):
    """Append measured column ownership layouts on compilers that support them."""
    if nt:
        return ()
    try:
        from triton.experimental.gluon.language.hcu import WHOLE_PACKET_LINEAR_READ_VERSION
    except ImportError:
        return ()
    required = ("buffer_store_scalar_offset", "buffer_store_split_immediate",
                "buffer_store_preserve_masked_base")
    from triton.backends.hcu.compiler import HIPOptions
    if (WHOLE_PACKET_LINEAR_READ_VERSION < 1
            or not all(name in HIPOptions.__dataclass_fields__ for name in required)
            or "large_overlap_group1_wait_fold" not in _codegen_packet_variants(nt=False)):
        return ()
    return (tuple(_COLUMN_PACKET_MODES) + ("wide_column_chart", "wide_column_overlap")
            + tuple(_COLUMN_VECTOR_OFFSET_MODES))


# NT cube geometries append to the complete existing family sequence. All
# option-dependent modes have distinct positional WAIT_MODE/GROUP_M constexprs.
_NT_CUBE_MODES = {
    "nt_cube_fullwait": (8, 0),
    "nt_cube_post_mma": (8, 1),
    "nt_cube_group1_post_mma": (1, 1),
}
_NT_CUBE_SOURCE_BASIS = (1, 2, 4, 8, 16, 32, 256, 512, 1024, 64, 128, 2048)
_NT_CUBE_OPTIONS = dict(
    num_warps=4, num_stages=5, waves_per_eu=1,
    loop_unroll_stage="post-pipeline", schedule_hint="local-prefetch",
    lds_dma_alias_workaround=True,
    mmac_transposed=True, mmac_warps_m=4,
    packet_matrix_read_partition=True, packet_wait_rotation=False,
    packet_operand_overlap_diagnostic=False, packet_reuse_wait="none",
    rebase_buffer_store=False,
)


def _nt_cube_packet_variants(*, nt):
    """Offer only available generic NT modes; preserve every old candidate."""
    if not nt or not _packet_supported(storage_layout=True):
        return ()
    try:
        from triton.backends.hcu.compiler import HIPOptions
    except ImportError:
        return ()
    fields = HIPOptions.__dataclass_fields__
    required = tuple(_NT_CUBE_OPTIONS) + (
        "mma_phase_schedule", "packet_post_mma_wait", "packet_redundant_lgkm_cleanup")
    if any(name not in fields for name in required):
        return ()
    modes = ["nt_cube_fullwait"]
    if "rect_post_mma" in _post_mma_packet_variants(nt=True):
        modes.extend(("nt_cube_post_mma", "nt_cube_group1_post_mma"))
    return tuple(modes)


_NT_LARGE_K16_OPTIONS = dict(
    _NN_GEOMETRY_OPTIONS, num_warps=8,
    coalesce_lgkm_waits=True, combine_shifted_disjoint_or=True,
    buffer_store_scalar_offset=True, buffer_store_split_immediate=True,
    buffer_store_preserve_masked_base=True,
)


def _nt_large_packet_variants(*, nt):
    """A full packet per operand shares B across a larger output row tile."""
    if not nt or not _packet_supported(storage_layout=True):
        return ()
    from triton.backends.hcu.compiler import HIPOptions
    if any(name not in HIPOptions.__dataclass_fields__
           for name in _NT_LARGE_K16_OPTIONS):
        return ()
    return ("nt_large_k16",)


@dataclass(frozen=True)
class _PacketProfile:
    # Runtime shapes never enter this registry. Geometry and code-generation
    # options describe the entire existing synchronous candidate family.
    geometry: tuple
    options: object


@lru_cache(maxsize=None)
def _synchronous_packet_profile(variant, *, nt):
    """Resolve one measured candidate to geometry, layouts and compiler options.

    ENTRY_FAMILY/WAIT_MODE/GROUP_M remain source constexprs because LibEntry
    does not include compiler-option kwargs in its cache key.
    """
    full_row = _NT_CUBE_SOURCE_BASIS
    prefix_row = full_row[:-1] + (0,)
    group = 8
    tiles = (2, 2)
    wait = 0
    entry = 0
    output = 0
    options = dict(_NN_GEOMETRY_OPTIONS)
    tile = None
    warps = (2, 2)
    a_basis = prefix_row
    b_basis = full_row
    nn_only = False
    nt_only = False
    if variant in ("large_safe", "large_overlap", "large_overlap_group1"):
        nn_only = True
        bm, bn, bk, wm, wn, a_basis, b_basis = _LARGE_PACKET_GEOMETRY
        tile, warps = (bm, bn, bk), (wm, wn)
        group = 1 if variant == "large_overlap_group1" else 8
        entry = 0 if variant == "large_safe" else 1
        options.update(num_warps=8, packet_post_mma_wait=variant == "large_safe",
                       mma_phase_schedule=True,
                       packet_operand_overlap_diagnostic=variant != "large_safe")
    elif variant in _CODEGEN_LARGE_MODES:
        nn_only = True
        bm, bn, bk, wm, wn, a_basis, b_basis = _LARGE_PACKET_GEOMETRY
        tile, warps = (bm, bn, bk), (wm, wn)
        group, entry, fold = _CODEGEN_LARGE_MODES[variant]
        options = dict(_CODEGEN_LARGE_OPTIONS, combine_shifted_disjoint_or=fold)
    elif variant in _COLUMN_PACKET_MODES or variant in _COLUMN_VECTOR_OFFSET_MODES:
        nn_only = True
        bm, bn, bk, wm, wn, a_basis, b_basis = _LARGE_PACKET_GEOMETRY
        tile, warps = (bm, bn, bk), (wm, wn)
        vector = variant in _COLUMN_VECTOR_OFFSET_MODES
        modes = _COLUMN_VECTOR_OFFSET_MODES if vector else _COLUMN_PACKET_MODES
        group, entry, output = modes[variant], 4 if vector else 3, 2
        options = dict(_CODEGEN_LARGE_OPTIONS, combine_shifted_disjoint_or=True,
                       buffer_store_scalar_offset=not vector,
                       buffer_store_split_immediate=True,
                       buffer_store_preserve_masked_base=True)
    elif variant == _CODEGEN_SQUARE_LABEL:
        nn_only = True
        tile = (128, 128, 16)
        b_basis = (1, 2, 4, 8, 16, 32, 64, 128, 512, 256, 1024, 0)
        output = 1
        options = dict(_CODEGEN_SQUARE_OPTIONS)
    elif variant in ("wide_column_chart", "wide_column_overlap"):
        nn_only = True
        config = _NN_GEOMETRIES["wide"]
        tile, warps = config['tile'], config['warps']
        a_basis, b_basis = config['A_SOURCE_BASIS'], config['B_SOURCE_BASIS']
        tiles, output = config['tiles_per_warp'], 2
        options.update(coalesce_lgkm_waits=True, combine_shifted_disjoint_or=True,
                       buffer_store_scalar_offset=True, buffer_store_split_immediate=True,
                       buffer_store_preserve_masked_base=True)
        if variant == "wide_column_overlap":
            wait = 4
            options.update(mma_phase_schedule=True, packet_operand_overlap_diagnostic=True,
                           packet_redundant_lgkm_cleanup=True)
    elif variant in _NT_CUBE_MODES:
        nt_only = True
        tile, a_basis = (64, 64, 64), full_row
        group, wait = _NT_CUBE_MODES[variant]
        options = dict(_NT_CUBE_OPTIONS, mma_phase_schedule=bool(wait),
                       packet_post_mma_wait=bool(wait),
                       packet_redundant_lgkm_cleanup=bool(wait))
    elif variant == "nt_large_k16":
        nt_only = True
        tile, warps, a_basis, group = (256, 256, 16), (4, 2), full_row, 1
        options = dict(_NT_LARGE_K16_OPTIONS)
    else:
        wait = int(variant.endswith("_post_mma"))
        geometry = variant.removesuffix("_post_mma") if wait else variant
        if geometry in _NN_GEOMETRIES:
            nn_only = True
            config = _NN_GEOMETRIES[geometry]
            tile, warps = config['tile'], config['warps']
            a_basis, b_basis = config['A_SOURCE_BASIS'], config['B_SOURCE_BASIS']
            tiles = config['tiles_per_warp']
        elif geometry in ("rect", "small", "square"):
            tile = {'rect': (64, 128, 32), 'small': (32, 64, 64),
                    'square': (128, 128, 32)}[geometry]
            if geometry == "small":
                tiles = (1, 2)
            if geometry == "square":
                a_basis = full_row
            if not nt:
                b_basis = (_NN_GEOMETRIES['cube']['B_SOURCE_BASIS'] if geometry == 'small'
                           else (1, 2, 4, 8, 16, 32, 64, 256, 1024, 128, 2048, 512))
        else:
            raise ValueError("Unsupported synchronous packet candidate")
        options.update(mma_phase_schedule=bool(wait), packet_post_mma_wait=bool(wait),
                       packet_redundant_lgkm_cleanup=bool(wait))
    if (nt and nn_only) or (not nt and nt_only):
        raise ValueError("Packet candidate does not support this B storage layout")
    options.update(GROUP_M=group, TILES_PER_WARP=tiles, WAIT_MODE=wait,
                   ENTRY_FAMILY=entry, OUTPUT_LAYOUT=output)
    return _PacketProfile((*tile, *warps, a_basis, b_basis), MappingProxyType(options))


def _launch_packet(a, b, c, *, variant="rect"):
    if variant == "splitu_nt_multi_row_k256":
        return _launch_multi_row_nt_k256_packet(a, b, c)
    if variant == "splitu_nt":
        return _launch_splitu_nt_packet(a, b, c)
    if variant in (
        "splitu_nt_k256", "splitu_nt_k256_peeled",
        "splitu_nt_k256_16", "splitu_nt_k256_peeled_16",
        "splitu_nt_k256_rotated", "splitu_nt_k256_rotated_16",
    ):
        return _launch_splitu_nt_k256_packet(
            a, b, c, peel_tail="_peeled" in variant,
            tile=16 if variant.endswith("_16") else 32,
            rotate_inputs="_rotated" in variant)
    if variant in ("raw_nt", "raw_nt_wgm4_c4"):
        return _launch_raw_nt_packet(a, b, c, variant=variant)
    m, k = a.shape
    n = b.shape[1]
    nt = b.stride() == (1, k)
    if (a.stride() != (k, 1) or (not nt and b.stride() != (n, 1))
            or c.stride() != (n, 1) or b.shape[0] != k or tuple(c.shape) != (m, n)):
        raise ValueError("Packet candidate requires contiguous A/C and NN or NT B")
    profile = _synchronous_packet_profile(variant, nt=nt)
    bm, bn = profile.geometry[:2]
    from .mm_packet_kernel import _mm_packet_rect_kernel
    return _mm_packet_rect_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        *profile.geometry, **profile.options)


def clear_mm_packet_choices():
    """Discard measured family choices, without clearing either tuning pool."""
    with _choice_lock:
        _choices.clear()
        _choice_measurements.clear()


def _packet_variants(*, nt, k=None):
    """One ordered capability-filtered registry for the measured MM families."""
    if k is not None and k >= _STRUCTURED_STRIDE_LIMIT:
        # These local-split entries use byte-addressed loads. The other
        # entries require a structured descriptor for at least one operand.
        return (_splitu_nt_packet_variants() + _multi_row_nt_packet_variants()) if nt else ()
    return (
            ("rect", "small", "square") + _raw_nt_packet_variants() + _splitu_nt_packet_variants() + _multi_row_nt_packet_variants()
            if nt
            else ("rect", "small", "square", "wide", "cube", "cube_tpw2") + _large_packet_variants()
        ) + _post_mma_packet_variants(nt=nt) + _codegen_packet_variants(nt=nt) + _nt_cube_packet_variants(nt=nt) + _column_packet_variants(nt=nt) + _nt_large_packet_variants(nt=nt)


def try_mm_packet(a, b, c, ordinary, *, tuning_key, ordinary_with_output=None,
                  variants=None):
    """Write c with the measured winner; return False outside the packet domain.

    ``ordinary`` tunes and launches the existing MM pool unchanged. Only family
    indices and scalar timing records are cached; tensors are never retained.
    Close finalists also use fresh outputs when the ordinary launcher supports
    them, matching the allocation behavior of the public MM entry point.
    Graph capture requires a public warmup, as for the ordinary autotuned route.
    A cold capture skips family comparison and retains the ordinary route.
    """
    if not _packet_eligible(a, b, c):
        return False
    nt = b.stride() == (1, a.shape[1])
    if variants is None:
        variants = _packet_variants(nt=nt, k=a.shape[1])
    else:
        variants = tuple(variants)
        if not nt or not variants or any(
            variant not in _splitu_nt_packet_variants() for variant in variants
        ):
            return False
    # Include effective pool/protocol and available family labels. Cache only
    # an integer index; later calls always launch with the current tensors.
    key = (
        a.device, tuple(a.shape), tuple(b.shape),
        tuple(a.stride()), tuple(b.stride()), tuple(c.stride()), tuning_key,
        ("packet-family-allocating-finalists-v1", variants,
         ordinary_with_output is not None),
    )
    with torch_device_fn.device(a.device):
        choice = _choices.get(key)
        if choice is None:
            if torch.cuda.is_current_stream_capturing():
                return False
            with _choice_lock:
                choice = _choices.get(key)
                if choice is None:
                    from triton.flagtune.runtime.graph_benchmark import (
                        do_bench_hcu_graph,
                    )

                    # Resolve the ordinary Default/Expanded policy before the
                    # comparison, including its existing persistent tune cache.
                    # Its candidate list and LibEntry ABI stay intact.
                    ordinary()
                    calls = (ordinary,) + tuple(
                        lambda variant=variant: _launch_packet(a, b, c, variant=variant)
                        for variant in variants
                    )
                    for call in calls[1:]:
                        call()
                    samples = tuple([] for _ in calls)
                    for repeat in range(3):
                        indices = range(len(calls))
                        for index in (indices if repeat % 2 == 0 else reversed(indices)):
                            samples[index].append(float(do_bench_hcu_graph(
                                calls[index], rep=10, n_retries=3,
                                device_interface=torch.cuda,
                            )))
                    medians = tuple(statistics.median(x) for x in samples)
                    choice = min(range(len(calls)), key=medians.__getitem__)
                    finalists = tuple(
                        index for index in sorted(range(len(calls)), key=medians.__getitem__)
                        if medians[index] <= medians[choice] * 1.05
                    )[:3]
                    refined = {}
                    if ordinary_with_output is not None and len(finalists) > 1:
                        # Short fixed-output screening can reverse a small
                        # difference once the public caller allocates its C.
                        # Refine only close candidates with the public replay
                        # duration, preserving all original tuning pools.
                        refined_samples = {index: [] for index in finalists}
                        for repeat in range(3):
                            order = finalists if repeat % 2 == 0 else reversed(finalists)
                            for index in order:
                                def fresh_output():
                                    output = torch.empty_like(c)
                                    if index:
                                        _launch_packet(a, b, output, variant=variants[index - 1])
                                    else:
                                        ordinary_with_output(output)
                                    return output

                                refined_samples[index].append(float(do_bench_hcu_graph(
                                    fresh_output, rep=100, n_retries=10,
                                    device_interface=torch.cuda,
                                )))
                        refined = {
                            index: statistics.median(values)
                            for index, values in refined_samples.items()
                        }
                        choice = min(finalists, key=refined.__getitem__)
                    _choice_measurements[key] = dict(
                        variants=("ordinary",) + variants,
                        screened_ms=medians, refined_allocating_ms=refined,
                        choice=choice,
                    )
                    _choices[key] = choice
                    logger.debug("Hygon MM family choice=%s screening=%s refined=%s ms",
                                 choice, medians, refined)
        if choice:
            _launch_packet(a, b, c, variant=variants[choice - 1])
        else:
            ordinary()
    return True
