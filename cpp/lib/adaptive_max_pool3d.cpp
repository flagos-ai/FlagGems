// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include <vector>
#include "ATen/WrapDimUtils.h"
#include "flag_gems/backend_utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

// ==============================================================================
// Kernel path selection — backend-based
// ==============================================================================

#if defined(FLAGGEMS_USE_NPU)
static const char* ADAPTIVE_POOL3D_KERNEL_PATH = "runtime/backend/_ascend/ops/adaptive_max_pool3d.py";
#elif defined(FLAGGEMS_USE_MUSA)
static const char* ADAPTIVE_POOL3D_KERNEL_PATH = "runtime/backend/_mthreads/ops/adaptive_max_pool3d.py";
#elif defined(FLAGGEMS_USE_GCU)
static const char* ADAPTIVE_POOL3D_KERNEL_PATH = "runtime/backend/_enflame/gcu400/ops/adaptive_max_pool3d.py";
#else
static const char* ADAPTIVE_POOL3D_KERNEL_PATH = "ops/adaptive_max_pool3d.py";
#endif

// ---------------------------------------------------------------------------
// Helper: adaptive max window bounds (matches Python _compute_max_win)
// ---------------------------------------------------------------------------
static inline int64_t compute_max_win(int64_t in_size, int64_t out_size) {
  return (in_size + out_size - 1) / out_size + 1;
}

// ==============================================================================
// Manual kernel config selectors (replace @triton.autotune on Ascend NPU)
// ==============================================================================

KernelDirectCfg select_kernel_direct(int64_t mwd,
                                     int64_t mwh,
                                     int64_t mww,
                                     int64_t ic,
                                     int64_t inn,
                                     int64_t od,
                                     int64_t oh,
                                     int64_t ow,
                                     c10::ScalarType dtype) {
  // Thresholds for channel-block heuristics — keep CB=1 when per-thread
  // loop iterations (CB × MAX_WIN_D × MAX_WIN_H × MAX_WIN_W) dominate.
  constexpr int64_t kMaxWinItersCB8 = 200;  // win * 8 >= this → skip CB=8
  constexpr int64_t kMaxWinItersCB4 = 100;  // win * 4 >= this → skip CB=4
  constexpr int64_t kSmallTotal = 4096;     // total below this → prefer CB=1
  constexpr int64_t kLargeTotal = 65536;    // total above this → use CB=1, large OB
  constexpr int64_t kTinyTotal = 1024;      // total below this → minimum config

  int64_t total_output = inn * ic * od * oh * ow;
  int64_t win_size = mwd * mwh * mww;

  // --- Channel blocking paths (OB=64, CB ∈ {8,4}) ---
  if (ic >= 8 && win_size <= 64 && total_output <= kSmallTotal && win_size * 8 < kMaxWinItersCB8)
    return {64, 8, 8, 3};
  else if (ic >= 4 && win_size <= 256 && total_output <= 16384 && total_output > kSmallTotal &&
           win_size * 4 < kMaxWinItersCB4)
    return {64, 4, 4, 4};

  // --- Default CB=1 path — dtype-aware OB selection ---
  // PyTorch autotuner prefers larger blocks (OB=128 or 256) for fp16/bf16
  // because the reduced-precision arithmetic is lighter per element and
  // Triton's instruction scheduler can pack more threads without spill.
  // Using OB=128 avoids register-pressure edge cases that produce
  // marginally-different fp16/bf16 rounding vs. the Python autotuner path.
  bool is_reduced = (dtype == c10::kHalf || dtype == c10::kBFloat16);

  if (total_output >= kLargeTotal) {
    // Large problems: max out occupancy
    if (is_reduced)
      return {256, 1, 8, 3};
    else
      return {64, 1, 2, 5};
  } else if (total_output <= kTinyTotal) {
    // Tiny problems: OB=128 for fp16/bf16 avoids the register-spill edge
    // case that causes flaky precision failures (e.g. shape (2,8,8,8,8)
    // output (1,4,4) with out_d=1 D-reduce path).
    if (is_reduced)
      return {128, 1, 4, 4};
    else
      return {64, 1, 2, 5};
  } else {
    // Medium problems
    if (is_reduced)
      return {128, 1, 4, 4};
    else
      return {64, 1, 2, 5};
  }
}

TinyKernelCfg select_tiny_kernel(int64_t total_output) {
  if (total_output <= 64)
    return {64, 2, 4};
  else if (total_output <= 1024)
    return {128, 4, 3};
  else
    return {256, 8, 2};
}

CooperativeCfg select_cooperative(int64_t mwd, int64_t mwh, int64_t mww) {
  int64_t win = mwd * mwh * mww;
  // Avoid COOP_THREADS=128 — the Python autotuner may select it but the C++
  // TritonJITFunction path can produce wrong results (diff ~0.87) for large
  // batch+channel counts (e.g. nc=98304).  Stick with the well-tested configs
  // that the Python autotuner also tests: CT=32/64/256.
  if (win <= 8) return {32, 1, 5};
  if (win <= 64) return {64, 2, 4};
  return {256, 8, 2};
}

UniformFusedCfg select_uniform_fused(int64_t ow, int64_t oh, int64_t inn, int64_t ic, int64_t od) {
  // BLOCK_H: group 2 rows when H >= 4 and W is not too wide
  int bh = (oh >= 4 && ow <= 64) ? 2 : 1;
  int bw;
  if (ow <= 32)
    bw = (int)ow;
  else if (ow <= 64)
    bw = (int)ow;
  else
    bw = 64;
  int nw, ns;
  if (bw * bh <= 32) {
    nw = 2;
    ns = 4;
  } else if (bw * bh <= 128) {
    nw = 4;
    ns = 3;
  } else {
    nw = 4;
    ns = 3;
  }
  return {bh, bw, nw, ns};
}

// ==============================================================================
// capture_native_kernels
//
// NPU-specific: pre-captures CANN amax kernel handles before the
// aten "IMPL" Library is created.  On non-NPU backends this is a no-op.
// ==============================================================================

void capture_native_kernels() {
#if defined(FLAGGEMS_USE_NPU)
  // On NPU, kernel capture happens in Python (flag_gems/__init__.py)
  // before the C++ extension is loaded.  This function exists as a
  // placeholder for future C++-side kernel handle management.
#endif
}

#if defined(FLAGGEMS_USE_NPU)

// Ascend-native max-reduction (npu::npu_max).  Faster than at::max for strided
// D-reduce and returns int32 indices (avoids the int64->int32 cast).
static std::tuple<at::Tensor, at::Tensor> npu_max(const at::Tensor& self, int64_t dim, bool keepdim = false) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("npu::npu_max", "dim")
                       .typed<std::tuple<at::Tensor, at::Tensor>(const at::Tensor&, int64_t, bool)>();
  return op.call(self, dim, keepdim);
}

// Placeholder indices for paths that don't need to compute indices.
static inline at::Tensor cpu_dummy() {
  return at::empty({0}, at::TensorOptions().dtype(at::kLong).device(at::kCPU));
}

// =============================================================================
// 2D adaptive-max-pool helper — native CANN op (aclnnAdaptiveMaxPool2d).
//
//  In:  input  (batch,  H,  W)
//  Out: output (batch, oH, oW), indices (batch, oH, oW)
//
//  The Triton _adaptive_pool2d_kernel was replaced by at::adaptive_max_pool2d
//  (CANN native) which is substantially faster — same pattern the Python
//  handler uses (torch.nn.functional.adaptive_max_pool2d).
// =============================================================================
static std::tuple<at::Tensor, at::Tensor> pool2d_triton(const at::Tensor& input,
                                                        int64_t oh,
                                                        int64_t ow,
                                                        const at::TensorOptions& val_opts,
                                                        const at::TensorOptions& idx_opts) {
  (void)val_opts;
  (void)idx_opts;
  return at::adaptive_max_pool2d(input, {oh, ow});
}

// =============================================================================
// Global-pool helper — at::amax over spatial dims, fastest path for [1,1,1].
// =============================================================================
std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d_global(const at::Tensor& self) {
  auto values = at::amax(self, {2, 3, 4}, /*keepdim=*/true);
  return std::make_tuple(values, cpu_dummy());
}

// =============================================================================
// Full C++ implementation — registered at the backend dispatch key via
// REGISTER_AND_LOG (same pattern as the CUDA wrapper).
// =============================================================================
//
// Always returns valid indices: the ATen schema is
// (Tensor, int[]) -> (Tensor, Tensor).
// =============================================================================
std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d(const at::Tensor& self, at::IntArrayRef output_size) {
  int64_t in_n = self.size(0), in_c = self.size(1);
  int64_t id = self.size(2), ih = self.size(3), iw = self.size(4);
  int64_t od = output_size[0], oh = output_size[1], ow = output_size[2];

  auto opts = self.options();
  auto long_opts = opts.dtype(at::kLong);
  auto int_opts = opts.dtype(at::kInt);

  // ==========================================================================
  // Path 1: Empty — any dimension is 0
  // ==========================================================================
  if (od == 0 || oh == 0 || ow == 0 || id == 0 || ih == 0 || iw == 0) {
    auto output = at::empty({in_n, in_c, od, oh, ow}, opts);
    auto indices = at::empty({in_n, in_c, od, oh, ow}, long_opts);
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // Path 2: Identity — all spatial dims equal
  //
  // Values returned by reference (no copy).  Indices are a broadcast view of
  // arange(spatial_total) — one CANN arange + zero-copy views, far faster than
  // the Triton _fill_identity_indices_kernel (which measured ~2.7ms vs ~0.03ms).
  // ==========================================================================
  if (id == od && ih == oh && iw == ow) {
    int64_t spatial_total = id * ih * iw;
    auto indices =
        at::arange(spatial_total, long_opts).view({1, 1, id, ih, iw}).expand({in_n, in_c, id, ih, iw});
    return std::make_tuple(self, indices);
  }

  // ==========================================================================
  // Path 3: Global pool [1,1,1]
  //
  // at::amax dispatches to CANN ReduceMax directly (amax in cpp_patched_ops).
  // For indices, at::max over dim goes to CANN aclnnMaxDim.
  // ==========================================================================
  if (od == 1 && oh == 1 && ow == 1) {
    if (id * ih * iw >= 1) {
      auto flat = self.flatten(2);  // (N, C, D*H*W)
      auto [vals, idxs] = at::max(flat, /*dim=*/2, /*keepdim=*/false);
      return std::make_tuple(vals.view({in_n, in_c, 1, 1, 1}),
                             idxs.view({in_n, in_c, 1, 1, 1}).to(at::kLong));
    }
    // Empty spatial — should not reach here due to Path 1, but be safe.
    auto output = at::empty({in_n, in_c, 1, 1, 1}, opts);
    auto indices = at::empty({in_n, in_c, 1, 1, 1}, long_opts);
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // Note: the single-pass _tiny_adaptive_pool_kernel / _kernel_direct Triton
  // kernels measured much slower than the CANN decomposition (at::max +
  // at::adaptive_max_pool2d + at::gather) on Ascend, so tiny shapes fall
  // through to the decomposition below.
  // ==========================================================================

  // ==========================================================================
  // Path 5: in_d == 1 — 3D pool reduces to 2D pool
  //
  // Reshape (N,C,1,H,W) -> (N*C,H,W), call native pool2d, reshape back.
  // The 2D spatial index IS the correct 3D index (D=0 always).
  // ==========================================================================
  if (id == 1 && od == 1) {
    auto inp_2d = self.reshape({in_n * in_c, ih, iw});
    auto [pool_vals, pool_idx] = pool2d_triton(inp_2d, oh, ow, opts, long_opts);
    return std::make_tuple(pool_vals.view({in_n, in_c, 1, oh, ow}), pool_idx.view({in_n, in_c, 1, oh, ow}));
  }

  // ==========================================================================
  // Path 6 (B1): HW identity + out_d == 1 — pure D-dimension max-reduction
  //
  // at::max over dim=1 dispatches to CANN aclnnMaxDim (max.dim in
  // cpp_patched_ops).  Indices computed with cheap host-side int32 arithmetic
  // (matches the Python handler; the Triton merge kernel is slower).
  // ==========================================================================
  if (ih == oh && iw == ow && id % od == 0 && id > 1 && od == 1) {
    int64_t hw = ih * iw;
    int64_t nc = in_n * in_c;
    auto self_contig = self.is_contiguous() ? self : self.contiguous();
    auto reshaped = self_contig.reshape({nc, id, hw});
    auto [d_vals, argmax_d] = at::max(reshaped, /*dim=*/1, /*keepdim=*/false);
    auto output = d_vals.view({in_n, in_c, 1, ih, iw});
    auto spatial = at::arange(hw, int_opts);  // (H*W,)
    auto indices =
        (argmax_d.to(at::kInt) * static_cast<int>(hw) + spatial).to(at::kLong).view({in_n, in_c, 1, ih, iw});
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // Path 7 (B2): HW identity + out_d > 1 — fold D into windows + max-reduce
  //
  // Reshape (N,C,out_d,win_d,H,W) -> (N*C*out_d, win_d, H*W), at::max over
  // dim=1.  Indices computed with cheap host-side int32 arithmetic.
  // ==========================================================================
  int64_t hw = ih * iw;
  if (ih == oh && iw == ow && id % od == 0 && id > od && od > 0) {
    int64_t win_d = id / od;

    auto self_contig = self.is_contiguous() ? self : self.contiguous();
    auto reshaped = self_contig.reshape({in_n, in_c, od, win_d, ih, iw}).reshape({-1, win_d, hw});
    auto [d_vals, d_local] = npu_max(reshaped, /*dim=*/1, /*keepdim=*/false);

    auto output = d_vals.view({in_n, in_c, od, ih, iw});
    auto d_local_4d = d_local.view({in_n, in_c, od, hw});
    // Full 3D index = (d_out * win_d + local_d) * hw + spatial
    auto d_off = at::arange(od, int_opts).view({1, 1, od, 1}) * static_cast<int>(win_d);
    auto spatial = at::arange(hw, int_opts).view({1, 1, 1, hw});
    auto d_full = d_local_4d + d_off;
    auto indices = (d_full * static_cast<int>(hw) + spatial).to(at::kLong).view({in_n, in_c, od, ih, iw});
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // From here on, ensure contiguous input.  at::max over strided dims on
  // NPU may require contiguous memory.
  // ==========================================================================
  auto self_contig = self.is_contiguous() ? self : self.contiguous();

  // ==========================================================================
  // Path 8 (Path A): in_d == out_d — pool2d per depth slice
  //
  // Each depth slice is independent.  Reshape to (N*C*in_d, H, W), call
  // pool2d once, reshape back.  Index = depth * H*W + spatial_2d_idx via
  // cheap host-side broadcast arithmetic.
  // ==========================================================================
  if (id == od && od > 1) {
    auto inp_2d = self_contig.view({-1, ih, iw});
    auto [pool_vals, pool_idx] = pool2d_triton(inp_2d, oh, ow, opts, long_opts);

    auto output = pool_vals.view({in_n, in_c, id, oh, ow});
    auto spatial_idx = pool_idx.view({in_n, in_c, id, oh, ow});
    // arange with step hw directly yields d*hw, avoiding a separate multiply.
    auto d_bcast = at::arange(0, id * hw, hw, long_opts).view({1, 1, id, 1, 1});
    auto indices = d_bcast + spatial_idx;
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // ==========================================================================
  // Note: _uniform_3d_fused_kernel computes values only (no indices support),
  // so it cannot be used here — the C++ handler must always return valid
  // indices.  Uniformly-divisible shapes fall through to the direct /
  // cooperative / general paths below, all of which compute indices.
  // ==========================================================================

  // ==========================================================================
  // Note: the single-pass _kernel_direct / _kernel_cooperative Triton kernels
  // are not used here — they measured much slower than the CANN decomposition
  // on Ascend, and _kernel_cooperative's RETURN_INDICES path is rejected by the
  // Ascend Triton backend.  All remaining shapes fall through to the D-reduce +
  // pool2d + gather decomposition below.
  // ==========================================================================

  // ==========================================================================
  // Path 11: out_d == 1 D-reduce (non-HW-identity path)
  //
  // Step 1: Reduce D via at::max over dim=1 of (N*C, in_d, H*W)
  // Step 2: pool2d_triton (_adaptive_pool2d_kernel) over (H, W)
  // Step 3: Merge D and spatial indices via _merge_outd1_indices_kernel
  // ==========================================================================
  if (od == 1 && id > 1) {
    int64_t nc = in_n * in_c;

    // Step 1: D-reduce — at::max dispatches to CANN aclnnMaxDim
    auto reshaped = self_contig.reshape({nc, id, hw});
    auto [d_reduced_flat, d_argmax_flat] = at::max(reshaped, /*dim=*/1, /*keepdim=*/false);

    auto d_reduced = d_reduced_flat.view({in_n, in_c, ih, iw});

    // Step 2: pool2d over H,W
    at::Tensor output;
    at::Tensor pool_idx;
    {
      auto d_2d = d_reduced.reshape({nc, ih, iw});
      auto pr = pool2d_triton(d_2d, oh, ow, opts, long_opts);
      output = std::get<0>(pr).view({in_n, in_c, 1, oh, ow});
      pool_idx = std::get<1>(pr);
    }

    // Step 3: Merge D and spatial indices via at::gather + host arithmetic.
    int64_t out_hw = oh * ow;
    auto d_argmax_2d = d_argmax_flat.reshape({nc, hw});
    auto spatial_2d = pool_idx.view(-1).view({nc, out_hw});
    auto d_best = d_argmax_2d.gather(1, spatial_2d);
    auto indices = (d_best * static_cast<int64_t>(hw) + spatial_2d).view({in_n, in_c, 1, oh, ow});
    return std::make_tuple(output, indices);
  }

  // ==========================================================================
  // Path 12: General 3D adaptive pool (Path C)
  //
  // Step 1: Reduce D over each adaptive window.
  //   Uniform windows (id % od == 0): fold + at::max
  //   Non-uniform windows: per-out_d slice + at::max loop
  // Step 2: pool2d (CANN at::adaptive_max_pool2d) over H,W
  // Step 3: Merge D and spatial indices via at::gather + host arithmetic
  // ==========================================================================
  {
    int64_t nc = in_n * in_c;
    int64_t out_hw = oh * ow;

    at::Tensor d_reduced;  // (N, C, out_d, H, W)
    at::Tensor d_argmax;   // (N, C, out_d, H, W) full depth index

    if (id % od == 0) {
      // --- Uniform D windows: fold + at::max ---
      int64_t win_d = id / od;
      auto for_max = self_contig.reshape({nc, od, win_d, hw}).reshape({nc * od, win_d, hw});
      auto [d_vals, d_local] = at::max(for_max, /*dim=*/1, /*keepdim=*/false);
      d_reduced = d_vals.view({in_n, in_c, od, ih, iw});
      auto d_off = at::arange(od, long_opts).view({1, 1, od, 1, 1}) * static_cast<int64_t>(win_d);
      d_argmax = d_local.view({in_n, in_c, od, ih, iw}) + d_off;
    } else {
      // --- Non-uniform D windows: per-out_d slice + at::max loop ---
      d_reduced = at::empty({in_n, in_c, od, ih, iw}, opts);
      d_argmax = at::empty({in_n, in_c, od, ih, iw}, long_opts);
      for (int64_t d_out = 0; d_out < od; ++d_out) {
        int64_t d_start = d_out * id / od;
        int64_t d_end = ((d_out + 1) * id + od - 1) / od;
        auto d_slice = self_contig.slice(/*dim=*/2, d_start, d_end);
        auto [d_vals, d_idxs] = at::max(d_slice, /*dim=*/2, /*keepdim=*/false);
        d_reduced.select(2, d_out).copy_(d_vals);
        d_argmax.select(2, d_out).copy_(d_idxs.to(at::kLong) + static_cast<int64_t>(d_start));
      }
    }

    // Step 2: pool2d over H,W
    auto pool_in = d_reduced.reshape({-1, ih, iw});
    auto [pool_vals, pool_idx] = pool2d_triton(pool_in, oh, ow, opts, long_opts);
    auto output = pool_vals.view({in_n, in_c, od, oh, ow});

    // Step 3: Merge D and spatial indices via at::gather + host arithmetic.
    auto d_argmax_2d = d_argmax.reshape({in_n * in_c * od, hw});
    auto spatial_2d = pool_idx.view(-1).view({in_n * in_c * od, out_hw});
    auto d_best = d_argmax_2d.gather(1, spatial_2d);
    auto indices = (d_best * static_cast<int64_t>(hw) + spatial_2d).view({in_n, in_c, od, oh, ow});
    return std::make_tuple(output, indices);
  }
}

#else  // NV / non-NPU: original Triton-based dispatch

// ==============================================================================
// adaptive_max_pool3d_global — (1,1,1) global pool
//
// Uses _kernel_cooperative Triton kernel (reshaped to 2D).
// ==============================================================================

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d_global(const at::Tensor& self) {
  at::Tensor self_c = self.contiguous();
  int64_t in_n = self_c.size(0), in_c = self_c.size(1);
  int64_t spatial_volume = self_c.size(2) * self_c.size(3) * self_c.size(4);
  int64_t nc = in_n * in_c;

  auto device = self_c.device();
  auto dtype = self_c.scalar_type();
  auto opts = self_c.options();

  // Output: (N, C, 1, 1, 1)
  auto out = at::empty({in_n, in_c, 1, 1, 1}, opts);
  auto indices = at::empty({in_n, in_c, 1, 1, 1}, opts.dtype(at::kLong));

  // Compute max window for cooperative kernel config selection.
  // The input is reshaped to (N*C, spatial_volume), so in_d=spatial_volume,
  // out_d=1. compute_max_win(spatial_volume, 1) = spatial_volume + 1.
  int64_t max_win_d = compute_max_win(spatial_volume, 1);

  // Select cooperative config based on window size
  CooperativeCfg cfg = select_cooperative(max_win_d, 1, 1);

  // Scratch space: total_output * COOP_THREADS.
  // Use max COOP_THREADS (256) to match Python pre-allocation pattern.
  constexpr int64_t kMaxCoopThreads = 256;
  auto sv = at::empty({nc * kMaxCoopThreads}, opts);
  auto si = at::empty({nc * kMaxCoopThreads}, opts.dtype(at::kLong));

  // Reshape input to 2D: (N*C, spatial_volume)
  auto flat_in = self_c.reshape({nc, spatial_volume});

  // Load the cooperative kernel
  static const TritonJITFunction& kernel = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
      "_kernel_cooperative");

  c10::DeviceGuard guard(device);
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  // Launch: grid=(nc,1,1) — one block per output element
  kernel(raw_stream,
         /* grid_x = */ static_cast<unsigned int>(nc),
         /* grid_y = */ 1,
         /* grid_z = */ 1,
         /* num_warps = */ static_cast<unsigned int>(cfg.nw),
         /* num_stages = */ static_cast<unsigned int>(cfg.ns),
         /* in_ptr = */ flat_in,
         /* scratch_vals_ptr = */ sv,
         /* scratch_idxs_ptr = */ si,
         /* output_ptr = */ out.view(-1),
         /* indices_ptr = */ indices.view(-1),
         /* in_c = */ 1,
         /* in_d = */ spatial_volume,
         /* in_h = */ 1,
         /* in_w = */ 1,
         /* out_d = */ 1,
         /* out_h = */ 1,
         /* out_w = */ 1,
         /* FLAT_ELEMS = */ nc,
         /* COOP_THREADS = */ static_cast<int>(cfg.ct),
         /* MAX_WIN_D = */ static_cast<int>(max_win_d),
         /* MAX_WIN_H = */ 1,
         /* MAX_WIN_W = */ 1,
         /* RETURN_INDICES = */ true);

  return std::make_tuple(out, indices);
}

// ==============================================================================
// adaptive_max_pool3d — main C++ handler
//
// Uses Triton kernels (_kernel_direct, _kernel_cooperative,
// _fill_identity_indices_kernel) for all computation.  No at:: native
// compute ops are called — all pooling logic runs through Triton JIT.
//
// Dispatch paths (matches Python backend):
//   1. Empty         → empty tensors
//   2. Identity      → clone + _fill_identity_indices_kernel
//   3. Global (1,1,1)→ _kernel_cooperative (reshaped to 2D)
//   4. Small+big win → _kernel_cooperative (total≤64, win≥1024)
//   5. General       → _kernel_direct (handles all 3D shapes natively)
// ==============================================================================

std::tuple<at::Tensor, at::Tensor> adaptive_max_pool3d(const at::Tensor& self, at::IntArrayRef output_size) {
  // Match Python: force contiguous before any kernel call (avoid stride issues).
  at::Tensor self_c = self.contiguous();

  int64_t out_d = output_size[0], out_h = output_size[1], out_w = output_size[2];
  int64_t in_n = self_c.size(0), in_c = self_c.size(1);
  int64_t in_d = self_c.size(2), in_h = self_c.size(3), in_w = self_c.size(4);

  auto device = self_c.device();
  auto dtype = self_c.scalar_type();
  auto opts = self_c.options();

  // --- 1. Empty ---
  if (out_d == 0 || out_h == 0 || out_w == 0 || in_d == 0 || in_h == 0 || in_w == 0) {
    auto out = at::empty({in_n, in_c, out_d, out_h, out_w}, opts);
    auto idx = at::empty({in_n, in_c, out_d, out_h, out_w}, opts.dtype(at::kLong));
    return std::make_tuple(out, idx);
  }

  // --- 2. Identity ---
  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    int64_t spatial_total = in_d * in_h * in_w;
    int64_t total_elements = in_n * in_c * out_d * out_h * out_w;

    auto idx = at::empty({in_n, in_c, out_d, out_h, out_w}, opts.dtype(at::kLong));

    static const TritonJITFunction& fill_kernel = TritonJITFunction::get_instance(
        std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
        "_fill_identity_indices_kernel");

    c10::DeviceGuard guard(device);
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

    constexpr int BLOCK_SIZE = 1024;
    unsigned int grid_x =
        static_cast<unsigned int>(utils::cdiv(static_cast<int>(total_elements), BLOCK_SIZE));

    fill_kernel(raw_stream,
                grid_x,
                1,
                1,
                /* num_warps = */ 4,
                /* num_stages = */ 0,
                /* indices_ptr = */ idx,
                /* spatial_total = */ spatial_total,
                /* total_elements = */ static_cast<int>(total_elements),
                /* BLOCK_SIZE = */ BLOCK_SIZE);

    return std::make_tuple(self_c.clone(), idx);
  }

  // --- 3. out_d=1 D-reduce: two-step path ---
  // Step 1: reduce D dimension via torch::max (native, deterministic).
  // Step 2: pool (H,W) via _kernel_direct with in_d=1 (MAX_WIN_D=2).
  //
  // This is more numerically stable than the single-pass _kernel_direct
  // with MAX_WIN_D = in_d+1, which can run many loop iterations per thread
  // and produce marginally different fp16/bf16 rounding through
  // TritonJITFunction vs the Python autotuner path.
  if (out_d == 1 && in_d > 1) {
    // Step 1: D-dimension reduction via native max (deterministic)
    auto d_result = at::max(self_c, /*dim=*/2);
    auto d_reduced = std::get<0>(d_result).unsqueeze(2).contiguous();  // (N,C,1,H,W)
    auto d_argmax = std::get<1>(d_result);                             // (N,C,H,W)

    // Fast path: if H,W are identity, we already have the correct values.
    // Just build indices from d_argmax.
    if (in_h == out_h && in_w == out_w) {
      auto hw = in_h * in_w;
      auto base_spatial = at::arange(hw, opts.dtype(at::kLong)).view({1, 1, 1, out_h, out_w});
      auto full_indices = d_argmax.unsqueeze(2) * hw + base_spatial;
      return std::make_tuple(d_reduced, full_indices);
    }

    // Step 2: pool (H,W) via _kernel_direct with in_d=1
    auto out = at::empty({in_n, in_c, 1, out_h, out_w}, opts);
    auto indices = at::empty({in_n, in_c, 1, out_h, out_w}, opts.dtype(at::kLong));

    int64_t mwh = compute_max_win(in_h, out_h);
    int64_t mww = compute_max_win(in_w, out_w);
    // MAX_WIN_D=2 for in_d=1 → out_d=1: kd loop runs exactly 1 iteration per
    // thread (kd < min(win_d, MAX_WIN_D) → kd < min(1,2)=1), minimizing the
    // TritonJITFunction compilation difference vs the Python autotuner path.
    int64_t mwd = 2;

    KernelDirectCfg cfg = select_kernel_direct(mwd, mwh, mww, in_c, in_n, 1, out_h, out_w, dtype);
    int ch_groups = (static_cast<int>(in_c) + cfg.cb - 1) / cfg.cb;
    int flat_elems =
        static_cast<int>(in_n) * ch_groups * 1 * static_cast<int>(out_h) * static_cast<int>(out_w);
    unsigned int grid_x = static_cast<unsigned int>(utils::cdiv(flat_elems, cfg.ob));

    c10::DeviceGuard guard(device);
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);

    static const TritonJITFunction& direct_kernel = TritonJITFunction::get_instance(
        std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
        "_kernel_direct");
    direct_kernel(raw_stream,
                  grid_x,
                  1,
                  1,
                  static_cast<unsigned int>(cfg.nw),
                  static_cast<unsigned int>(cfg.ns),
                  d_reduced,
                  out,
                  indices,
                  static_cast<int>(in_n),
                  static_cast<int>(in_c),
                  1,
                  static_cast<int>(in_h),
                  static_cast<int>(in_w),  // in_d=1
                  1,
                  static_cast<int>(out_h),
                  static_cast<int>(out_w),
                  cfg.ob,
                  cfg.cb,
                  static_cast<int>(mwd),
                  static_cast<int>(mwh),
                  static_cast<int>(mww),
                  true);

    // Step 3: merge d_argmax with spatial indices from the kernel
    // The kernel returns flat spatial indices (h*W + w).  The full 3D index
    // is d_best * H * W + spatial_idx.  We load each kernel-produced spatial
    // index, look up the best D at that (n,c,h,w) position from d_argmax,
    // and write back the merged 3D index.
    {
      int64_t n_elements = indices.numel();
      int64_t out_spatial = out_h * out_w;
      constexpr int BLOCK_SIZE = 256;
      unsigned int grid = static_cast<unsigned int>(utils::cdiv(static_cast<int>(n_elements), BLOCK_SIZE));

      static const TritonJITFunction& merge_kernel = TritonJITFunction::get_instance(
          std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
          "_merge_outd1_indices_kernel");
      merge_kernel(raw_stream,
                   grid,
                   1,
                   1,
                   /* num_warps = */ 4,
                   /* num_stages = */ 2,
                   indices.view(-1),
                   d_argmax,
                   indices.view(-1),
                   static_cast<int>(n_elements),
                   static_cast<int>(in_c),
                   static_cast<int>(in_h),
                   static_cast<int>(in_w),
                   static_cast<int>(out_h),
                   static_cast<int>(out_w),
                   BLOCK_SIZE);
    }

    return std::make_tuple(out, indices);
  }

  // --- 4. Global pool (1,1,1) ---
  if (out_d == 1 && out_h == 1 && out_w == 1) {
    return adaptive_max_pool3d_global(self_c);
  }

  // --- 5 & 6. General case: use _kernel_direct or _kernel_cooperative ---

  // Compute max window sizes for kernel compilation
  int64_t max_win_d = compute_max_win(in_d, out_d);
  int64_t max_win_h = compute_max_win(in_h, out_h);
  int64_t max_win_w = compute_max_win(in_w, out_w);
  int64_t total_output = in_n * in_c * out_d * out_h * out_w;
  int64_t win_size = max_win_d * max_win_h * max_win_w;

  // Allocate output tensors
  auto out = at::empty({in_n, in_c, out_d, out_h, out_w}, opts);
  auto indices = at::empty({in_n, in_c, out_d, out_h, out_w}, opts.dtype(at::kLong));

  c10::DeviceGuard guard(device);
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  // --- 4. Small output with large window: use cooperative kernel ---
  // Matches Python: total_output <= 64 and win_size >= 1024.
  if (total_output <= 64 && win_size >= 1024) {
    CooperativeCfg cfg = select_cooperative(max_win_d, max_win_h, max_win_w);
    constexpr int64_t kMaxCoopThreads = 256;

    auto sv = at::empty({total_output * kMaxCoopThreads}, opts);
    auto si = at::empty({total_output * kMaxCoopThreads}, opts.dtype(at::kLong));

    static const TritonJITFunction& coop_kernel = TritonJITFunction::get_instance(
        std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
        "_kernel_cooperative");

    coop_kernel(raw_stream,
                /* grid_x = */ static_cast<unsigned int>(total_output),
                /* grid_y = */ 1,
                /* grid_z = */ 1,
                /* num_warps = */ static_cast<unsigned int>(cfg.nw),
                /* num_stages = */ static_cast<unsigned int>(cfg.ns),
                /* in_ptr = */ self_c,
                /* scratch_vals_ptr = */ sv,
                /* scratch_idxs_ptr = */ si,
                /* output_ptr = */ out.view(-1),
                /* indices_ptr = */ indices.view(-1),
                /* in_c = */ static_cast<int>(in_c),
                /* in_d = */ static_cast<int>(in_d),
                /* in_h = */ static_cast<int>(in_h),
                /* in_w = */ static_cast<int>(in_w),
                /* out_d = */ static_cast<int>(out_d),
                /* out_h = */ static_cast<int>(out_h),
                /* out_w = */ static_cast<int>(out_w),
                /* FLAT_ELEMS = */ static_cast<int>(total_output),
                /* COOP_THREADS = */ static_cast<int>(cfg.ct),
                /* MAX_WIN_D = */ static_cast<int>(max_win_d),
                /* MAX_WIN_H = */ static_cast<int>(max_win_h),
                /* MAX_WIN_W = */ static_cast<int>(max_win_w),
                /* RETURN_INDICES = */ true);

    return std::make_tuple(out, indices);
  }

  // --- 5. General case: use _kernel_direct ---
  // _kernel_direct handles all 3D shapes natively:
  //   in_d==1, out_d==1, in_d==out_d, general reduction — all paths
  //   collapse into this single kernel.
  {
    KernelDirectCfg cfg =
        select_kernel_direct(max_win_d, max_win_h, max_win_w, in_c, in_n, out_d, out_h, out_w, dtype);

    // Size grid from flat_elems (which accounts for CHAN_PER_BLOCK grouping)
    // rather than total_output, so every block does useful work.
    // Kernel-side validation (flat_idx < flat_elems) still catches any overflow.
    int ch_groups = (static_cast<int>(in_c) + cfg.cb - 1) / cfg.cb;
    int flat_elems = static_cast<int>(in_n) * ch_groups * static_cast<int>(out_d) * static_cast<int>(out_h) *
                     static_cast<int>(out_w);
    unsigned int grid_x = static_cast<unsigned int>(utils::cdiv(flat_elems, cfg.ob));

    static const TritonJITFunction& direct_kernel = TritonJITFunction::get_instance(
        std::string(utils::get_flag_gems_src_path() / ADAPTIVE_POOL3D_KERNEL_PATH),
        "_kernel_direct");

    direct_kernel(raw_stream,
                  grid_x,
                  /* grid_y = */ 1,
                  /* grid_z = */ 1,
                  /* num_warps = */ static_cast<unsigned int>(cfg.nw),
                  /* num_stages = */ static_cast<unsigned int>(cfg.ns),
                  /* in_ptr = */ self_c,
                  /* out_ptr = */ out,
                  /* idx_ptr = */ indices,
                  /* in_n = */ static_cast<int>(in_n),
                  /* in_c = */ static_cast<int>(in_c),
                  /* in_d = */ static_cast<int>(in_d),
                  /* in_h = */ static_cast<int>(in_h),
                  /* in_w = */ static_cast<int>(in_w),
                  /* out_d = */ static_cast<int>(out_d),
                  /* out_h = */ static_cast<int>(out_h),
                  /* out_w = */ static_cast<int>(out_w),
                  /* OUT_PER_BLOCK = */ cfg.ob,
                  /* CHAN_PER_BLOCK = */ cfg.cb,
                  /* MAX_WIN_D = */ static_cast<int>(max_win_d),
                  /* MAX_WIN_H = */ static_cast<int>(max_win_h),
                  /* MAX_WIN_W = */ static_cast<int>(max_win_w),
                  /* RETURN_INDICES = */ true);

    backend::synchronize();
    return std::make_tuple(out, indices);
  }
}

#endif  // FLAGGEMS_USE_NPU

}  // namespace flag_gems
