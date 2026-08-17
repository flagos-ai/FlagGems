import pytest
import torch

from . import base, consts

# ============================================================================
# Comprehensive (input_shape, output_size) pairs for adaptive_max_pool3d.
# ============================================================================

ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS = (
    # ========================================================================
    # 1. Path A — Identity (all spatial dims match)
    # ========================================================================
    ((2, 256, 2, 14, 14), (2, 14, 14)),  # exact identity
    ((2, 128, 8, 16, 16), (8, 16, 16)),  # exact identity, power-of-2
    ((2, 256, 4, 7, 7), (4, 7, 7)),  # exact identity, odd spatial
    # ========================================================================
    # 2. Path B — out_d=1 fast path: D-reduce first then 2D spatial pool.
    # ========================================================================
    # B, then downstream → various paths on reduced (N,C,1,H,W) tensor
    ((1, 8, 64, 256, 256), (1, 7, 7)),  # classic: extreme H/W 256→7
    ((1, 64, 8, 56, 56), (1, 56, 56)),  # T=8→1, H/W preserved
    ((1, 256, 64, 112, 112), (1, 112, 112)),  # T=64→1, spatial preserved
    # B → also global pool (1,1,1) after D-reduce:
    ((1, 4096, 64, 64, 64), (1, 1, 1)),  # B→C: reduced spatial=4096≥4096
    ((1, 8, 32, 128, 128), (1, 1, 1)),  # B→C: reduced spatial=16384≥4096
    ((1, 4, 8, 24, 32), (1, 1, 1)),  # B→D: reduced spatial=768∈[64,4096)
    ((1, 128, 16, 28, 28), (1, 1, 1)),  # B→D: reduced spatial=784∈[64,4096)
    # Real video model classifier heads (all B→E: reduced spatial=49<64)
    ((1, 1024, 8, 7, 7), (1, 1, 1)),  # I3D Mixed_5c
    ((1, 256, 32, 7, 7), (1, 1, 1)),  # SlowFast fast
    ((1, 2048, 4, 7, 7), (1, 1, 1)),  # SlowFast slow
    ((1, 256, 30, 7, 7), (1, 1, 1)),  # R3D-18 variant
    ((1, 256, 24, 7, 7), (1, 1, 1)),  # R3D-50 variant
    ((1, 256, 14, 7, 7), (1, 1, 1)),  # SlowFast variant
    ((1, 256, 10, 7, 7), (1, 1, 1)),  # SlowFast variant
    ((1, 4096, 4, 4, 4), (1, 1, 1)),  # B→E: huge C, reduced spatial=16<64
    ((1, 4096, 2, 4, 4), (1, 1, 1)),  # B→E: huge C, reduced spatial=16<64
    # B, aggressive spatial reduction (not global, out_d=1 but out_h,w > 1)
    ((8, 64, 16, 112, 112), (1, 7, 7)),  # win=4913, total=25088
    # ========================================================================
    # 3. Path C — Global torch.max (direct: in_d=1, spatial >= 4096)
    # ========================================================================
    ((1, 256, 1, 64, 64), (1, 1, 1)),  # spatial=4096 (boundary: >=4096)
    ((2, 128, 1, 128, 64), (1, 1, 1)),  # spatial=8192
    ((1, 512, 1, 56, 112), (1, 1, 1)),  # spatial=6272
    # ========================================================================
    # 4. Path D — Global block_reduce (direct: in_d=1, 64 <= spatial < 4096)
    # ========================================================================
    ((2, 512, 1, 8, 8), (1, 1, 1)),  # spatial=64 (boundary: >=64)
    ((1, 256, 1, 14, 14), (1, 1, 1)),  # spatial=196
    ((2, 128, 1, 7, 14), (1, 1, 1)),  # spatial=98
    ((1, 64, 1, 28, 28), (1, 1, 1)),  # spatial=784
    # ========================================================================
    # 5. Path E — Global→1D kernel (direct: in_d=1, spatial < 64)
    # ========================================================================
    ((2, 512, 1, 7, 7), (1, 1, 1)),  # R3D-18 head, spatial=49
    ((2, 2048, 1, 4, 4), (1, 1, 1)),  # R3D-50 head, spatial=16
    ((2, 512, 1, 4, 4), (1, 1, 1)),  # C3D fc6, spatial=16
    ((1, 1024, 1, 4, 4), (1, 1, 1)),  # spatial=16
    # ========================================================================
    # 6. Known model shapes — temporal compression (mostly Path G: 1D kernel)
    # ========================================================================
    ((1, 768, 160, 32, 32), (8, 32, 32)),  # PLLaVA T=160→8
    ((2, 1024, 32, 48, 48), (16, 48, 48)),  # VideoLLaMA T=32→16
    ((4, 128, 32, 64, 64), (8, 64, 64)),  # InternVideo T=32→8
    ((2, 256, 64, 40, 40), (16, 40, 40)),  # temporal compression
    # ========================================================================
    # 7. Known model shapes — full 3D compression (all dims reduced)
    # ========================================================================
    # Path F — Large window kernel
    ((1, 16, 64, 128, 128), (4, 8, 8)),  # win=4913, total=4096→F
    # Path G — 1D kernel (large total, moderate window ≤ 2048)
    ((1, 1280, 48, 36, 50), (8, 8, 8)),  # Qwen2.5-VL, win=336→G
    ((2, 64, 6, 16, 32), (2, 8, 16)),  # asymmetric, win=36→G
    ((2, 640, 64, 42, 72), (16, 14, 14)),  # Qwen2.5-VL moderate, win=140→G
    ((2, 768, 96, 32, 32), (16, 8, 8)),  # VideoLLaMA long-T, win=175→G
    ((2, 512, 64, 64, 64), (4, 8, 8)),  # medical 3D, win=1377→G
    # Path H — 2D fast (win > 2048, out_h,w ≤ 16)
    ((1, 8, 64, 256, 256), (64, 7, 7)),  # T keep, extreme H/W, win=2888→H
    # Path I — 2D regular (win > 2048, out_h>16 or out_w>16)
    ((2, 64, 64, 256, 256), (2, 32, 32)),  # win=2673>2048, out_h=32→I
    # ========================================================================
    # 8. Known model shapes — spatial compression (D preserved, H/W reduced)
    # ========================================================================
    ((2, 128, 4, 28, 28), (4, 14, 14)),  # D keep, H/W 28→14, win=18→G
    ((8, 64, 16, 56, 56), (16, 14, 14)),  # D keep, H/W 56→14, win=50→G
    # ========================================================================
    # 9. Path G — 1D kernel (small total, !prefer_2d)
    # ========================================================================
    ((1, 2, 4, 4, 4), (2, 2, 2)),  # total=16, win=27→G (very small)
    ((1, 4, 8, 16, 16), (4, 8, 8)),  # total=1024, win=27→G
    # ========================================================================
    # 10. Path G — 1D kernel (large total, win ≤ 2048)
    # ========================================================================
    ((4, 32, 16, 32, 32), (8, 16, 16)),  # total=262144, win=27→G
    ((2, 16, 8, 16, 16), (4, 8, 8)),  # total=8192, win=27→G
    # ========================================================================
    # 11. Path H — 2D fast via prefer_2d (total≤512, large N*C forces 2D)
    # ========================================================================
    ((16, 4, 4, 4, 4), (2, 2, 2)),  # total=512, N*C=64→prefer_2d→H
    ((8, 8, 4, 4, 4), (2, 2, 2)),  # total=512, N*C=64→prefer_2d→H
    # ========================================================================
    # 12. Edge cases — unit dims, non-divisible, stress
    # ========================================================================
    ((2, 256, 16, 1, 14), (8, 1, 14)),  # H=1, win=12→G
    ((2, 256, 16, 14, 1), (8, 14, 1)),  # W=1, win=12→G
    ((2, 256, 1, 1, 14), (1, 1, 7)),  # D=1, H=1, win=12→G
    ((1, 3, 16, 224, 224), (8, 7, 7)),  # video input, win=3267→F
    ((1, 64, 37, 59, 43), (7, 13, 19)),  # all prime, win=168→G
    ((1, 1, 5, 9, 11), (1, 1, 1)),  # C=1, tiny, out_d=1→B
    # ========================================================================
    # 13. Batch / channel size stress
    # ========================================================================
    ((128, 768, 4, 4, 4), (1, 1, 1)),  # huge batch, out_d=1→B
    ((32, 512, 8, 8, 8), (4, 4, 4)),  # large batch, total=1048576→G
    # ========================================================================
    # 14. Decision-boundary shapes (near heuristic thresholds)
    # ========================================================================
    # Near blocks_2d_est=156 (Path F ↔ G):
    ((1, 32, 64, 64, 64), (4, 4, 4)),  # blk_est=128<156→F (just below)
    ((1, 40, 64, 64, 64), (4, 4, 4)),  # blk_est=160≥156→G (just above)
    # Near win_size=2048 (1D ↔ 2D for total>4096):
    ((1, 32, 32, 128, 128), (4, 8, 8)),  # win=2601>2048, total=8192→H
    ((2, 64, 16, 64, 64), (2, 8, 8)),  # win=729≤2048, total=16384→G
    # Near total_output=4096:
    ((1, 16, 8, 8, 8), (4, 8, 8)),  # total=4096 exactly, win=12→G
    ((1, 8, 8, 8, 9), (4, 8, 8)),  # total=2048<4096, win=18→G
    # Near spatial=64 boundary (global D ↔ E, direct):
    ((1, 256, 1, 8, 8), (1, 1, 1)),  # spatial=64→D (boundary: ≥64)
    ((1, 256, 1, 7, 9), (1, 1, 1)),  # spatial=63→E (boundary: <64)
    # Near spatial=4096 boundary (global C ↔ D, direct):
    ((1, 128, 1, 64, 64), (1, 1, 1)),  # spatial=4096→C (boundary: ≥4096)
    ((1, 128, 1, 63, 65), (1, 1, 1)),  # spatial=4095→D (boundary: <4096)
)

# ============================================================================
# COMPREHENSIVE-ONLY shapes — additional variants for thorough coverage.
# Run with --bench-level comprehensive.
# ============================================================================
ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE = (
    # ========================================================================
    # C1: More global pooling variants
    # ========================================================================
    # Additional B→E (in_d>1, reduced spatial<64):
    ((1, 256, 20, 7, 7), (1, 1, 1)),  # C3D classifier variant
    ((2, 512, 8, 4, 4), (1, 1, 1)),  # C3D variant, spatial after B=16
    ((2, 512, 7, 4, 4), (1, 1, 1)),  # C3D variant
    ((2, 512, 6, 4, 4), (1, 1, 1)),  # C3D variant
    ((2, 512, 5, 4, 4), (1, 1, 1)),  # C3D variant
    ((2, 512, 3, 4, 4), (1, 1, 1)),  # C3D variant
    ((2, 512, 2, 4, 4), (1, 1, 1)),  # C3D variant
    ((2, 3, 7, 13, 17), (1, 1, 1)),  # B→D: reduced spatial=221
    # More direct D (in_d=1, 64≤spatial<4096):
    ((2, 512, 1, 32, 8), (1, 1, 1)),  # spatial=256→D
    ((1, 1024, 1, 16, 16), (1, 1, 1)),  # spatial=256→D
    # More direct C (in_d=1, spatial≥4096):
    ((2, 256, 1, 128, 128), (1, 1, 1)),  # spatial=16384→C
    ((1, 64, 1, 256, 256), (1, 1, 1)),  # spatial=65536→C
    # More direct E (in_d=1, spatial<64):
    ((2, 512, 1, 3, 4), (1, 1, 1)),  # spatial=12→E
    # ========================================================================
    # C2: More Path B variants (non-global out_d=1)
    # ========================================================================
    ((2, 512, 8, 56, 56), (1, 28, 28)),  # D-reduce then 2D spatial
    ((1, 128, 32, 48, 64), (1, 12, 16)),  # non-aligned after reduction
    ((2, 256, 16, 112, 112), (1, 56, 56)),  # large H/W after reduction
    # ========================================================================
    # C3: More temporal compression (D reduced, H/W preserved → mostly G)
    # ========================================================================
    ((1, 256, 64, 112, 112), (4, 112, 112)),  # T=64→4, win=68→G
    ((2, 768, 64, 40, 40), (32, 8, 8)),  # PLLaVA variant, win=108→G
    ((1, 512, 128, 28, 28), (16, 28, 28)),  # long temporal, win=36→G
    # ========================================================================
    # C4: More full 3D compression
    # ========================================================================
    ((2, 256, 2, 14, 14), (2, 7, 7)),  # small D, win=18→G
    ((1, 256, 32, 128, 128), (4, 8, 8)),  # medical 3D, win=2601→H (2D fast)
    ((1, 640, 64, 24, 40), (8, 8, 8)),  # Qwen scaled-down, win=216→G
    ((1, 8, 64, 128, 128), (4, 8, 8)),  # win=4913, total=2048→F
    # ========================================================================
    # C5: More spatial compression (D preserved, H/W reduced → mostly G)
    # ========================================================================
    ((1, 64, 8, 112, 112), (8, 56, 56)),  # win=18→G
    ((2, 256, 4, 56, 56), (4, 28, 28)),  # D preserved, win=18→G
    ((1, 3, 32, 224, 224), (16, 56, 56)),  # video pyramid, win=75→G
    # ========================================================================
    # C6: Small / odd / non-aligned edge cases
    # ========================================================================
    ((1, 16, 64, 107, 73), (7, 12, 18)),  # non-aligned, win=660→G
    ((1, 3, 5, 13, 17), (2, 5, 7)),  # small odd dims, win=64→G
    ((1, 1, 3, 7, 11), (1, 3, 5)),  # C=1, small odd, out_d=1→B
    # ========================================================================
    # C7: Path I — 2D regular (win > 2048, out_h>16 or out_w>16)
    # ========================================================================
    ((1, 16, 64, 256, 256), (2, 32, 32)),  # win=2673>2048, out_h=32→I
    ((1, 8, 128, 256, 256), (2, 17, 17)),  # win=18785>2048, out_h=17→I
    # ========================================================================
    # C8: prefer_2d boundary (total≤512, N*C marginal → 2D vs 1D flip)
    # ========================================================================
    ((8, 4, 4, 4, 4), (2, 2, 2)),  # total=512, N*C=32→prefer_2d→H
    ((2, 4, 4, 4, 4), (2, 2, 2)),  # total=128, N*C=8, grid1=1→prefer_2d→H
    ((1, 8, 4, 4, 4), (2, 2, 2)),  # total=64, N*C=8, grid1=1→prefer_2d→H
    ((2, 1, 4, 4, 4), (2, 2, 2)),  # total=64, N*C=2<4→!prefer_2d→G
    # ========================================================================
    # C9: Large window kernel boundary (blocks_2d_est crossing 156)
    # ========================================================================
    ((1, 36, 64, 64, 64), (4, 4, 4)),  # blk_est=144<156→F (below)
    ((1, 38, 64, 64, 64), (4, 4, 4)),  # blk_est=152<156→F (below)
    # ========================================================================
    # C10: 1D kernel with large inner loops
    # ========================================================================
    ((1, 64, 64, 64, 64), (4, 4, 4)),  # win=4913, total=4096→G (large loop!)
    ((1, 32, 128, 128, 128), (4, 4, 4)),  # win=35937, total=2048→F
    ((1, 16, 256, 256, 256), (4, 4, 4)),  # win=274625, total=1024→F
    ((1, 8, 64, 128, 256), (2, 4, 4)),  # win=70785, total=256→F
    ((8, 16, 16, 64, 64), (2, 8, 8)),  # win=729, total=16384→G (1D, moderate win)
    ((4, 32, 32, 56, 56), (4, 14, 14)),  # win=180, total=200704→G (1D, moderate win)
    # ========================================================================
    # C11: Tensor-core-friendly aligned shapes (multiples of 8)
    # ========================================================================
    ((2, 64, 16, 64, 64), (8, 16, 16)),  # aligned, win=75→G
    ((4, 128, 8, 32, 64), (4, 16, 32)),  # aligned, win=27→G
    ((1, 256, 32, 128, 128), (8, 16, 16)),  # aligned, win=405→G
    # ========================================================================
    # C12: Stress tests — extreme parameter values
    # ========================================================================
    ((1, 8192, 2, 2, 2), (1, 1, 1)),  # extreme C=8192, out_d=1→B
    ((256, 3, 8, 14, 14), (4, 7, 7)),  # extreme batch=256, win=27→G
    ((1, 1, 128, 256, 256), (8, 8, 8)),  # C=1, large spatial, win=18513→F
    ((1, 4096, 2, 2, 2), (1, 1, 1)),  # huge C, tiny spatial, out_d=1→B
    # ========================================================================
    # C13: Known open-source video model configs
    # ========================================================================
    ((1, 768, 96, 14, 14), (1, 1, 1)),  # TimeSformer ViT-B/16 @ 96f → B
    ((1, 768, 16, 14, 14), (1, 1, 1)),  # VideoMAE ViT-B @ 16f → B
    ((1, 768, 32, 7, 7), (1, 1, 1)),  # VideoSwin → B (reduced sp=49)
    ((2, 768, 16, 14, 14), (4, 7, 7)),  # MViT feature map, win=45→G
    # ========================================================================
    # C14: More identity / near-identity shapes
    # ========================================================================
    ((4, 64, 8, 32, 32), (8, 32, 32)),  # identity → A
    ((1, 128, 16, 28, 28), (16, 28, 28)),  # identity → A
    ((4, 3, 32, 112, 112), (32, 112, 112)),  # identity → A
    # ========================================================================
    # C15: Cubic output_size (integer → broadcast to (D,D,D) in operator)
    # ========================================================================
    ((1, 128, 32, 64, 64), (8, 8, 8)),  # cubic, win=405→G
    ((2, 256, 16, 48, 48), (4, 4, 4)),  # cubic, win=845→G
    ((1, 512, 64, 128, 128), (2, 2, 2)),  # cubic, win=139425→G
    # ========================================================================
    # C16: win_size near 2048 — more boundary variants
    # ========================================================================
    ((2, 32, 32, 96, 96), (4, 8, 8)),  # win=2028≤2048→G (just below)
    ((4, 16, 48, 64, 64), (4, 8, 8)),  # win=2025≤2048→G (just below)
    ((2, 8, 64, 96, 96), (2, 4, 4)),  # win=20625>2048→H
)


def _input_fn(shapes, dtype, device):
    """Yield (input_tensor, output_size) pair for each shape config."""
    input_shape, output_size = shapes
    inp = base.generate_tensor_input(input_shape, dtype, device)
    yield inp, output_size


# ============================================================================
# Core benchmark shapes — one representative set covering every dispatch path
# with a few sizes each, plus real video-model shapes.  Used at --level core.
# ============================================================================
ADAPTIVE_MAX_POOL3D_BENCH_QUICK_CONFIGS = (
    # --- Path A: identity ---
    ((2, 256, 2, 14, 14), (2, 14, 14)),
    ((2, 128, 8, 16, 16), (8, 16, 16)),
    # --- Global pool (1,1,1) ---
    ((1, 256, 1, 64, 64), (1, 1, 1)),
    ((1, 1024, 8, 7, 7), (1, 1, 1)),
    # --- Path B: out_d=1, D-reduce + 2D pool ---
    ((1, 8, 64, 256, 256), (1, 7, 7)),
    ((8, 64, 16, 112, 112), (1, 7, 7)),
    # --- B1: HW identity + out_d=1 ---
    ((1, 64, 8, 56, 56), (1, 56, 56)),
    ((1, 256, 64, 112, 112), (1, 112, 112)),
    # --- B2: HW identity + out_d>1 ---
    ((4, 128, 32, 64, 64), (8, 64, 64)),
    ((2, 1024, 32, 48, 48), (16, 48, 48)),
    # --- Path A: in_d == out_d ---
    ((2, 128, 4, 28, 28), (4, 14, 14)),
    ((8, 64, 16, 56, 56), (16, 14, 14)),
    # --- Path C: general uniform D ---
    ((2, 640, 64, 42, 72), (16, 14, 14)),
    ((1, 1280, 48, 36, 50), (8, 8, 8)),
    ((32, 512, 8, 8, 8), (4, 4, 4)),
    # --- Non-uniform D windows ---
    ((1, 3, 5, 6, 7), (2, 3, 4)),
    ((1, 3, 5, 13, 17), (2, 5, 7)),
    # --- Tiny shapes ---
    ((1, 2, 4, 4, 4), (2, 2, 2)),
    ((1, 4, 8, 16, 16), (4, 8, 8)),
    # --- Real video-model shapes ---
    ((1, 768, 160, 32, 32), (8, 32, 32)),
    ((1, 16, 64, 128, 128), (4, 8, 8)),
)


class AdaptiveMaxPool3dBenchmark(base.GenericBenchmark):
    def init_user_config(self):
        super().init_user_config()
        # Always use file-defined shapes at both core and comprehensive levels,
        # overriding the YAML core_shapes.yaml subset.
        self.shapes = ADAPTIVE_MAX_POOL3D_BENCH_QUICK_CONFIGS
        if base.Config.bench_level == consts.BenchLevel.COMPREHENSIVE:
            self.shapes = list(
                dict.fromkeys(
                    ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS
                    + ADAPTIVE_MAX_POOL3D_BENCH_CONFIGS_COMPREHENSIVE
                )
            )


# ============================================================================
# C++ wrapper performance test — torch.ops.flag_gems.* path (§3.1)
#
# The TORCH_LIBRARY_IMPL routes through the Python autotuner (same as
# c_operators), so a single C++ perf test covers both dispatch paths.
# ============================================================================


def _cpp_input_fn(shapes, dtype, device):
    """Yield (input_tensor, output_size_list) for the C++ op path.

    torch.ops.flag_gems.adaptive_max_pool3d requires output_size as a
    plain Python list (not a tuple).
    """
    input_shape, output_size = shapes
    inp = base.generate_tensor_input(input_shape, dtype, device)
    yield inp, list(output_size)


@pytest.mark.adaptive_max_pool3d
def test_perf_adaptive_max_pool3d():
    """flag_gems perf test — same shapes/iterations for both env modes.

    USE_C_EXTENSION=1 → gems_op = torch.ops.flag_gems.adaptive_max_pool3d (C++ handler)
    USE_C_EXTENSION=0 → gems_op = None → base falls back to flag_gems.use_gems()
                        (Python handler patching torch_op)
    Both are measured against the same native torch reference (torch_op) and
    the same ADAPTIVE_MAX_POOL3D_BENCH_QUICK_CONFIGS shapes.
    """
    from flag_gems.config import has_c_extension

    if has_c_extension:
        input_fn = _cpp_input_fn
        gems_op = torch.ops.flag_gems.adaptive_max_pool3d
    else:
        input_fn = _input_fn
        gems_op = None  # Python flaggems handler via flag_gems.use_gems()

    bench = AdaptiveMaxPool3dBenchmark(
        input_fn=input_fn,
        op_name="adaptive_max_pool3d",
        torch_op=torch.nn.functional.adaptive_max_pool3d,
        gems_op=gems_op,
        dtypes=consts.FLOAT_DTYPES,
    )
    bench.run()
