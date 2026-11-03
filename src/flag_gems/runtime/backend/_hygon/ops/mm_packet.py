# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Select a whole-packet kernel against the ordinary, fully tuned MM family."""

import logging
import statistics
import threading

import torch
import triton

from flag_gems.runtime import torch_device_fn

logger = logging.getLogger(__name__)
_choices = {}
_choice_measurements = {}
_choice_lock = threading.RLock()


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
        m > 1 and n > 1 and 256 <= k <= 4096
        and k & (k - 1) == 0 and (nt or n % 16 == 0)
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


def _launch_large_packet(a, b, c, *, variant):
    if variant not in ("large_safe", "large_overlap", "large_overlap_group1") or b.stride() == (1, a.shape[1]):
        raise ValueError("Large packet modes require eligible contiguous NN storage")
    from .mm_packet_large_kernel import _mm_packet_kernel
    m, k = a.shape
    n = b.shape[1]
    traversal = {"GROUP_M": 1} if variant == "large_overlap_group1" else {}
    return _mm_packet_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        # LibEntry keys signature constexprs, not compiler-option kwargs.
        # This unused constexpr separates the two measured native modes.
        ENTRY_FAMILY=0 if variant == "large_safe" else 1,
        num_warps=8, num_stages=5, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", schedule_hint="local-prefetch",
        lds_dma_alias_workaround=True,
        mmac_transposed=True, mmac_warps_m=4,
        packet_matrix_read_partition=True,
        mma_phase_schedule=True, packet_wait_rotation=False,
        packet_post_mma_wait=variant == "large_safe",
        packet_operand_overlap_diagnostic=variant in ("large_overlap", "large_overlap_group1"),
        packet_reuse_wait="none", packet_redundant_lgkm_cleanup=False,
        rebase_buffer_store=False,
        **traversal,
    )



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
    from .mm_splitu_nt_multi_row_k256_kernel import _small_nt_multi_row_k256

    m, k = a.shape
    n = b.shape[1]
    if (k < 256 or k % 256 or a.stride() != (k, 1)
            or b.stride() != (1, k) or c.stride() != (n, 1)):
        raise ValueError("Multi-row K256 requires contiguous A/C and transposed B")
    return _small_nt_multi_row_k256[(triton.cdiv(m, 80), triton.cdiv(n, 32), 1)](
        a, b.t(), c, m, n, k, 16, 32, 5,
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
    from .mm_splitu_nt_kernel import _small_nt_splitu
    m, k = a.shape
    n = b.shape[1]
    if b.stride() != (1, k):
        raise ValueError("Split-U NT requires contiguous transposed B storage")
    return _small_nt_splitu[(triton.cdiv(m, 16), triton.cdiv(n, 32), 1)](
        a, b, c, m, n, k, 16, 32,
        PADDED=True, ROW_WARP_STRIDE=1, DOUBLE_BUFFER=True,
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
    schedule = {}
    if rotate_inputs:
        from .mm_splitu_nt_k256_rotated_kernel import _small_nt_splitu_k256_rotated

        if not _loop_carried_refill_supported():
            raise ValueError("Rotated inputs require loop-carried refill support")
        kernel = _small_nt_splitu_k256_rotated
        schedule["loop_carried_store_refill"] = True
    elif peel_tail:
        from .mm_splitu_nt_k256_peeled_kernel import _small_nt_splitu_k256_peeled

        kernel = _small_nt_splitu_k256_peeled
    else:
        from .mm_splitu_nt_k256_kernel import _small_nt_splitu_k256

        kernel = _small_nt_splitu_k256

    m, k = a.shape
    n = b.shape[1]
    if (k < 256 or k % 256 or a.stride() != (k, 1)
            or b.stride() != (1, k) or c.stride() != (n, 1)):
        raise ValueError("K256 NT requires contiguous A/C and transposed contiguous B")
    if tile not in (16, 32):
        raise ValueError("Unsupported K256 NT output tile")
    return kernel[(triton.cdiv(m, tile), triton.cdiv(n, tile), 1)](
        a, b.t(), c, m, n, k, tile, tile,
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


def _packet_wait_options(wait_mode):
    # Compiler kwargs do not enter LibEntry's key. The source constexpr also
    # distinguishes these native modes, without changing the runtime pointers.
    return dict(WAIT_MODE=1, mma_phase_schedule=True, packet_post_mma_wait=True,
                packet_wait_rotation=False, packet_redundant_lgkm_cleanup=True) if wait_mode else {}


def _launch_nn_geometry_packet(a,b,c,*,geometry,wait_mode=False):
    config=_NN_GEOMETRIES[geometry];m,k=a.shape;n=b.shape[1]
    if a.stride()!=(k,1)or b.stride()!=(n,1)or c.stride()!=(n,1):
        raise ValueError('Wide/cube packet modes require contiguous NN storage')
    if b.shape[0]!=k or tuple(c.shape)!=(m,n):raise ValueError('Mismatched matrix dimensions')
    from .mm_packet_kernel import _mm_packet_rect_kernel
    fn = _mm_packet_rect_kernel
    bm,bn,bk=config['tile'];wm,wn=config['warps']
    options = dict(_NN_GEOMETRY_OPTIONS, **_packet_wait_options(wait_mode))
    return fn[((m+bm-1)//bm*((n+bn-1)//bn),)](
        a,b,c,m,n,k,*a.stride(),*b.stride(),*c.stride(),bm,bn,bk,wm,wn,
        config['A_SOURCE_BASIS'],config['B_SOURCE_BASIS'],
        TILES_PER_WARP=config['tiles_per_warp'],**options,
    )


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


def _codegen_nn_dimensions(a, b, c):
    m, k = a.shape
    n = b.shape[1]
    if (a.stride() != (k, 1) or b.stride() != (n, 1)
            or c.stride() != (n, 1)):
        raise ValueError("Code-generation packet modes require contiguous NN storage")
    if b.shape[0] != k or tuple(c.shape) != (m, n):
        raise ValueError("Mismatched matrix dimensions")
    return m, n, k


def _launch_codegen_square_packet(a, b, c):
    """The measured compact square tile with local output byte offsets."""
    from .mm_packet_tilebase_kernel import _mm_packet_rect_kernel
    m, n, k = _codegen_nn_dimensions(a, b, c)
    return _mm_packet_rect_kernel[(triton.cdiv(m, 128) * triton.cdiv(n, 128),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        128, 128, 16, 2, 2,
        (1, 2, 4, 8, 16, 32, 256, 512, 1024, 64, 128, 0),
        (1, 2, 4, 8, 16, 32, 64, 128, 512, 256, 1024, 0),
        GROUP_M=8, TILES_PER_WARP=(2, 2), WAIT_MODE=0, ENTRY_FAMILY=0,
        **_CODEGEN_SQUARE_OPTIONS,
    )


def _launch_codegen_large_packet(a, b, c, *, variant):
    from .mm_packet_large_kernel import _mm_packet_kernel
    group_m, entry_family, fold = _CODEGEN_LARGE_MODES[variant]
    m, n, k = _codegen_nn_dimensions(a, b, c)
    return _mm_packet_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        GROUP_M=group_m, ENTRY_FAMILY=entry_family,
        combine_shifted_disjoint_or=fold, **_CODEGEN_LARGE_OPTIONS,
    )


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


def _launch_wide_column_packet(a, b, c, *, overlap=False):
    """Apply the compiler-derived column chart to the existing wide geometry."""
    from .mm_packet_column_rect_kernel import _mm_packet_rect_kernel
    config = _NN_GEOMETRIES["wide"]
    m, n, k = _codegen_nn_dimensions(a, b, c)
    bm, bn, bk = config["tile"]
    wm, wn = config["warps"]
    options = dict(_NN_GEOMETRY_OPTIONS, coalesce_lgkm_waits=True,
                   combine_shifted_disjoint_or=True,
                   buffer_store_scalar_offset=True,
                   buffer_store_split_immediate=True,
                   buffer_store_preserve_masked_base=True)
    if overlap:
        # As for large_overlap, partial-LGKM hardware ordering remains an
        # explicitly unproved boundary. Keep the original measured family.
        # WAIT_MODE distinguishes LibEntry keys from compiler-option kwargs.
        options.update(WAIT_MODE=4, mma_phase_schedule=True,
                       packet_operand_overlap_diagnostic=True,
                       packet_redundant_lgkm_cleanup=True)
    return _mm_packet_rect_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        bm, bn, bk, wm, wn, config["A_SOURCE_BASIS"], config["B_SOURCE_BASIS"],
        TILES_PER_WARP=config["tiles_per_warp"], **options,
    )


def _launch_column_packet(a, b, c, *, variant):
    """Select the adjacent output chart; retain all original memory-domain gates."""
    from .mm_packet_column_kernel import _mm_packet_kernel
    m, n, k = _codegen_nn_dimensions(a, b, c)
    vector_offsets = variant in _COLUMN_VECTOR_OFFSET_MODES
    modes = _COLUMN_VECTOR_OFFSET_MODES if vector_offsets else _COLUMN_PACKET_MODES
    return _mm_packet_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        GROUP_M=modes[variant], ENTRY_FAMILY=4 if vector_offsets else 3,
        combine_shifted_disjoint_or=True,
        buffer_store_scalar_offset=not vector_offsets, buffer_store_split_immediate=True,
        buffer_store_preserve_masked_base=True,
        **_CODEGEN_LARGE_OPTIONS,
    )


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


def _launch_nt_cube_packet(a, b, c, *, variant):
    """Existing NT22 ABI with the measured full 4096-element producer charts."""
    if variant not in _NT_CUBE_MODES:
        raise ValueError("Unsupported NT cube packet mode")
    m, k = a.shape
    n = b.shape[1]
    if (a.stride() != (k, 1) or b.stride() != (1, k)
            or c.stride() != (n, 1) or b.shape[0] != k
            or tuple(c.shape) != (m, n)):
        raise ValueError("NT cube requires contiguous A/C and transposed contiguous B")
    from .mm_packet_kernel import _mm_packet_rect_kernel
    group_m, wait_mode = _NT_CUBE_MODES[variant]
    options = dict(
        _NT_CUBE_OPTIONS, mma_phase_schedule=bool(wait_mode),
        packet_post_mma_wait=bool(wait_mode),
        packet_redundant_lgkm_cleanup=bool(wait_mode),
    )
    return _mm_packet_rect_kernel[(triton.cdiv(m, 64) * triton.cdiv(n, 64),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        64, 64, 64, 2, 2, _NT_CUBE_SOURCE_BASIS, _NT_CUBE_SOURCE_BASIS,
        group_m, (2, 2), wait_mode, **options,
    )


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


def _launch_nt_large_k16_packet(a, b, c):
    from .mm_packet_kernel import _mm_packet_rect_kernel

    m, k = a.shape
    n = b.shape[1]
    if (a.stride() != (k, 1) or b.stride() != (1, k)
            or c.stride() != (n, 1) or b.shape[0] != k
            or tuple(c.shape) != (m, n)):
        raise ValueError("Large NT packet requires contiguous A/C and transposed B")
    # Explicit geometry and traversal constexprs also distinguish LibEntry keys.
    return _mm_packet_rect_kernel[(triton.cdiv(m, 256) * triton.cdiv(n, 256),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        256, 256, 16, 4, 2, _NT_CUBE_SOURCE_BASIS, _NT_CUBE_SOURCE_BASIS,
        1, (2, 2), 0, **_NT_LARGE_K16_OPTIONS,
    )


def _launch_packet(a, b, c, *, variant="rect"):
    if variant == "nt_large_k16":
        return _launch_nt_large_k16_packet(a, b, c)
    if variant == "splitu_nt_multi_row_k256":
        return _launch_multi_row_nt_k256_packet(a, b, c)
    if variant in ("wide_column_chart", "wide_column_overlap"):
        return _launch_wide_column_packet(
            a, b, c, overlap=variant == "wide_column_overlap")
    if variant in _COLUMN_PACKET_MODES or variant in _COLUMN_VECTOR_OFFSET_MODES:
        return _launch_column_packet(a, b, c, variant=variant)
    if variant in _NT_CUBE_MODES:
        return _launch_nt_cube_packet(a, b, c, variant=variant)
    if variant == _CODEGEN_SQUARE_LABEL:
        return _launch_codegen_square_packet(a, b, c)
    if variant in _CODEGEN_LARGE_MODES:
        return _launch_codegen_large_packet(a, b, c, variant=variant)
    wait_mode = variant in (
        "rect_post_mma", "small_post_mma", "square_post_mma",
        "wide_post_mma", "cube_post_mma", "cube_tpw2_post_mma")
    if wait_mode:
        variant = variant.removesuffix("_post_mma")
    if variant in ("wide", "cube", "cube_tpw2"):
        return _launch_nn_geometry_packet(a, b, c, geometry=variant, wait_mode=wait_mode)
    if variant in (
        "splitu_nt_k256", "splitu_nt_k256_peeled",
        "splitu_nt_k256_16", "splitu_nt_k256_peeled_16",
        "splitu_nt_k256_rotated", "splitu_nt_k256_rotated_16",
    ):
        # Tail peeling helps some tile occupancies and costs others a few
        # registers. Keep both in the existing measured family comparison.
        return _launch_splitu_nt_k256_packet(
            a, b, c, peel_tail="_peeled" in variant,
            tile=16 if variant.endswith("_16") else 32,
            rotate_inputs="_rotated" in variant,
        )
    if variant == "splitu_nt":
        return _launch_splitu_nt_packet(a, b, c)
    if variant in ("raw_nt", "raw_nt_wgm4_c4"):
        return _launch_raw_nt_packet(a, b, c, variant=variant)
    if variant in ("large_safe", "large_overlap", "large_overlap_group1"):
        return _launch_large_packet(a, b, c, variant=variant)
    m, k = a.shape
    n = b.shape[1]
    nt = b.stride() == (1, k)
    small = variant == "small"
    square = variant == "square"
    if variant not in ("rect", "small", "square"):
        raise ValueError("Unsupported packet tile/layout combination")
    bm, bn, bk = ((32, 64, 64) if small else
                  (128, 128, 32) if square else (64, 128, 32))
    from .mm_packet_kernel import _mm_packet_rect_kernel
    b_source_basis = (
        (1, 2, 4, 8, 16, 32, 256, 512, 1024, 64, 128, 2048)
        if nt else (1, 2, 4, 8, 16, 32, 128, 512, 1024, 64, 256, 2048)
        if small else
        (1, 2, 4, 8, 16, 32, 64, 256, 1024, 128, 2048, 512)
    )
    layout_options = ({"TILES_PER_WARP": (1, 2)} if small else
                      {"TILES_PER_WARP": (2, 2)} if square else {})
    if wait_mode:
        layout_options["WAIT_MODE"] = 1
    # The square A tile uses the full 4096-element chart; existing prefix
    # tiles retain their original zero final basis. No shape-specific route.
    a_source_basis = (1, 2, 4, 8, 16, 32, 256, 512, 1024, 64, 128,
                      2048 if square else 0)
    return _mm_packet_rect_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, b, c, m, n, k, *a.stride(), *b.stride(), *c.stride(),
        bm, bn, bk, 2, 2,
        a_source_basis,
        b_source_basis,
        num_warps=4, num_stages=5, waves_per_eu=1,
        loop_unroll_stage="post-pipeline", schedule_hint="local-prefetch",
        lds_dma_alias_workaround=True,
        mmac_transposed=True, mmac_warps_m=4,
        packet_matrix_read_partition=True,
        mma_phase_schedule=wait_mode, packet_wait_rotation=False,
        packet_post_mma_wait=wait_mode, packet_operand_overlap_diagnostic=False,
        packet_reuse_wait="none", packet_redundant_lgkm_cleanup=wait_mode,
        rebase_buffer_store=False,
        **layout_options,
    )


def clear_mm_packet_choices():
    """Discard measured family choices, without clearing either tuning pool."""
    with _choice_lock:
        _choices.clear()
        _choice_measurements.clear()


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
        variants = (
            ("rect", "small", "square") + _raw_nt_packet_variants() + _splitu_nt_packet_variants() + _multi_row_nt_packet_variants()
            if nt
            else ("rect", "small", "square", "wide", "cube", "cube_tpw2") + _large_packet_variants()
        ) + _post_mma_packet_variants(nt=nt) + _codegen_packet_variants(nt=nt) + _nt_cube_packet_variants(nt=nt) + _column_packet_variants(nt=nt) + _nt_large_packet_variants(nt=nt)
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
