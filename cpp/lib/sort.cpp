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

#include "flag_gems/backend_utils.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "ATen/WrapDimUtils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

#if defined(FLAGGEMS_USE_MUSA)
static const char* SORT_KERNEL_PATH = "runtime/backend/_mthreads/ops/sort.py";
#elif defined(FLAGGEMS_USE_GCU)
static const char* SORT_KERNEL_PATH = "runtime/backend/_enflame/gcu400/ops/sort.py";
#elif defined(FLAGGEMS_USE_MLU)
static const char* SORT_KERNEL_PATH = "runtime/backend/_cambricon/ops/sort.py";
#else
static const char* SORT_KERNEL_PATH = "ops/sort.py";
#endif

int64_t get_num_bits(const at::ScalarType& dtype) {
  if (dtype == torch::kBool) {
    return 1;
  }
  return c10::elementSize(dtype) * 8;
}

#if defined(FLAGGEMS_USE_MLU)
// The generic sweep kernel resolves its decoupled look-back by spinning on the
// status of pid_n - 1, which needs cross-CTA forward-progress guarantees the MLU
// scheduler does not provide: it silently returns unsorted data for n above one
// CTA tile and eventually faults the queue. The Cambricon kernels keep pid_n as
// a real grid dimension and fold the batch into a per-program loop instead, so
// the launch geometry below mirrors radix_sort() in
// runtime/backend/_cambricon/ops/sort.py.
static std::tuple<at::Tensor, at::Tensor> radix_sort_mlu(const at::Tensor& arr,
                                                         int64_t k_bits,
                                                         bool descending) {
  int64_t n = arr.size(-1);
  int64_t m = arr.numel() / n;
  TORCH_CHECK(n < (1 << 30), "we have not implemented 2**30 per launch");

  const int64_t num_bits = get_num_bits(arr.scalar_type());
  const int64_t num_bins = 1 << k_bits;
  const int64_t n_passes = utils::cdiv(num_bits, k_bits);

  // These kernels must run as single-core (BLOCK) tasks: the sweep spins on the
  // status word written by pid_n - 1, and the multi-core UNION task type the
  // runtime selects for num_warps > 1 deadlocks that look-back. num_warps=1 and
  // num_stages=0 are also what the Triton MLU backend picks by default for the
  // Python driver these kernels were written for.
  constexpr unsigned int NUM_WARPS = 1;
  constexpr unsigned int NUM_STAGES = 0;
  // The MLU runtime turns grid_x into grid_x * num_warps tasks before checking
  // it against the hardware limit, so budget for that here.
  constexpr int64_t MLU_MAX_GRID_X = 65535;
  const int64_t max_grid_x = MLU_MAX_GRID_X / NUM_WARPS;

  const int64_t TILE_N_HIST = 512;
  const int64_t TILES_N_PER_CTA_HIST = 8;
  const int64_t CTA_TILE_N_HIST = TILES_N_PER_CTA_HIST * TILE_N_HIST;
  const int64_t TILE_R_HIST = 16;

  const int64_t grid_n_hist = utils::cdiv(n, CTA_TILE_N_HIST);
  const int64_t total_tasks_hist = m * grid_n_hist;
  const unsigned int grid_x_hist = static_cast<unsigned int>(std::min(total_tasks_hist, max_grid_x));

  c10::DeviceGuard guard(arr.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  at::Tensor global_hist =
      at::zeros({m, n_passes, num_bins}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));

  const TritonJITFunction& hist_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                      "compute_global_hist_kernel");
  hist_kernel(raw_stream,
              grid_x_hist,
              1,
              1,
              NUM_WARPS,
              NUM_STAGES,
              arr,
              global_hist,
              n_passes,
              m,
              n,
              grid_n_hist,
              total_tasks_hist,
              TILES_N_PER_CTA_HIST,
              TILE_N_HIST,
              TILE_R_HIST,
              k_bits,
              descending);

  // NOTE: no cast back to int32 here. at::cumsum promotes the int32 histogram
  // to int64, and the Cambricon sweep kernel is compiled against that i64
  // buffer - narrowing it changes the element size the kernel loads with and
  // silently produces garbage positions.
  at::Tensor ex_cumsum_bins = at::cumsum(global_hist, -1) - global_hist;

  at::Tensor arr_in = arr.clone();
  at::Tensor indices_in = at::arange(0, n, at::TensorOptions().dtype(torch::kInt64).device(arr.device()))
                              .broadcast_to(arr.sizes())
                              .contiguous();
  at::Tensor arr_out = at::empty_like(arr_in);
  at::Tensor indices_out = at::empty_like(indices_in);

  const int64_t TILE_R_SWEEP = 8;
  const int64_t TILE_N_SWEEP = 3072;
  const int64_t grid_r_sweep = utils::cdiv(num_bins, TILE_R_SWEEP);
  const int64_t grid_n_sweep = utils::cdiv(n, TILE_N_SWEEP);
  // Split the batch so that grid_x stays within the hardware limit; each
  // program then walks M_PER_SPLIT rows itself.
  const int64_t splits = std::max<int64_t>(1, utils::cdiv(m, max_grid_x));
  const int64_t m_per_split = utils::cdiv(m, splits);

  at::Tensor status =
      at::empty({m, num_bins, grid_n_sweep}, at::TensorOptions().device(arr.device()).dtype(torch::kUInt32));

  const TritonJITFunction& sweep_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                      "sweep");

  for (int64_t i = 0; i < n_passes; ++i) {
    int64_t bit_offset = i * k_bits;
    status.zero_();
    for (int64_t pid_n_base = 0; pid_n_base < grid_n_sweep; pid_n_base += max_grid_x) {
      const int64_t grid_n_chunk = std::min(max_grid_x, grid_n_sweep - pid_n_base);
      sweep_kernel(raw_stream,
                   static_cast<unsigned int>(splits),
                   static_cast<unsigned int>(grid_n_chunk),
                   static_cast<unsigned int>(grid_r_sweep),
                   NUM_WARPS,
                   NUM_STAGES,
                   arr_in,
                   indices_in,
                   arr_out,
                   indices_out,
                   ex_cumsum_bins,
                   status,
                   n_passes,
                   i,
                   bit_offset,
                   m,
                   n,
                   grid_n_sweep,
                   pid_n_base,
                   TILE_N_SWEEP,
                   TILE_R_SWEEP,
                   k_bits,
                   descending,
                   m_per_split);
    }

    std::swap(arr_in, arr_out);
    std::swap(indices_in, indices_out);
  }

  return std::make_tuple(arr_in, indices_in);
}
#endif  // FLAGGEMS_USE_MLU

std::tuple<at::Tensor, at::Tensor> radix_sort(const at::Tensor& arr, int64_t k_bits, bool descending) {
#if defined(FLAGGEMS_USE_MLU)
  return radix_sort_mlu(arr, k_bits, descending);
#else
  int64_t n = arr.size(-1);
  int32_t m = arr.numel() / n;
  TORCH_CHECK(n < (1 << 30), "we have not implemented 2**30 per launch");
#if defined(FLAGGEMS_USE_MUSA)
  // Match Python Triton launches: runtime shape/loop scalars are i32.  An
  // i64 ABI here needlessly widens address arithmetic throughout sweep.
  const int32_t n_i32 = static_cast<int32_t>(n);
  const int32_t m_i32 = m;
#endif

  auto dtype = arr.scalar_type();
  int64_t num_bits = get_num_bits(dtype);

  const int64_t TILE_N_HIST = 1024;
  const int64_t TILES_N_PER_CTA_HIST = 8;
  const int64_t CTA_TILE_N_HIST = TILES_N_PER_CTA_HIST * TILE_N_HIST;

  const int64_t num_bins = 1 << k_bits;
  const int64_t n_passes = utils::cdiv(num_bits, k_bits);
  const int64_t TILE_R_HIST = 16;
#if defined(FLAGGEMS_USE_MUSA)
  const bool use_compact_indices = m >= 512 && n <= 2048;
  const int64_t grid_n_hier = utils::cdiv(n, 2048);
  const bool use_hierarchical = !use_compact_indices && num_bins == 16 && grid_n_hier >= 2 &&
                                (m >= 1024 || (grid_n_hier >= 4 && m >= 64));
#endif

  int64_t grid_n_hist = utils::cdiv(n, CTA_TILE_N_HIST);
#if defined(FLAGGEMS_USE_GCU)
  unsigned int grid_x_hist = std::min((int64_t)(m * grid_n_hist), (int64_t)48);
#else
  unsigned int grid_x_hist = m * grid_n_hist;
#endif

  const TritonJITFunction& hist_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                      "compute_global_hist_kernel");

  c10::DeviceGuard guard(arr.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  at::Tensor ex_cumsum_bins;
#if defined(FLAGGEMS_USE_MUSA)
  if (use_hierarchical) {
    // Filled one pass at a time from tile_hist below; this avoids an
    // additional all-passes read of the payload before hierarchical scatter.
    ex_cumsum_bins =
        at::empty({m, n_passes, num_bins}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));
  } else {
#endif
    at::Tensor global_hist =
        at::zeros({m, n_passes, num_bins}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));
    hist_kernel(raw_stream,
                grid_x_hist,
                1,
                1,
                4,
                1,
                arr,
                global_hist,
#if defined(FLAGGEMS_USE_MUSA)
                static_cast<int32_t>(n_passes),
                m_i32,
                n_i32,
#else
              n_passes,
              m,
              n,
#endif
#if defined(FLAGGEMS_USE_GCU)
                grid_n_hist,
#endif
                TILES_N_PER_CTA_HIST,
                TILE_N_HIST,
                TILE_R_HIST,
                k_bits,
                descending);
    ex_cumsum_bins = (at::cumsum(global_hist, -1) - global_hist).to(torch::kInt32);
#if defined(FLAGGEMS_USE_MUSA)
  }
#endif

#if defined(FLAGGEMS_USE_MUSA)
  // Long rows are dominated by the decoupled-lookback portion of sweep.  For
  // these shapes, replace the per-bin CAS/spin chain with a compact
  // per-tile histogram, a metadata-only prefix, and one scatter program per
  // tile.  Keep the existing OneSweep path for short rows where the extra
  // histogram/scan launches are more expensive than lookback.
  const int64_t TILE_N_HIER = 2048;
  if (use_hierarchical) {
    at::Tensor arr_in = arr;
    at::Tensor arr_out = at::empty_like(arr);
    at::Tensor arr_scratch = at::empty_like(arr);
    at::Tensor indices_in = at::empty_like(arr, arr.options().dtype(torch::kInt64));
    at::Tensor indices_out = at::empty_like(arr, arr.options().dtype(torch::kInt64));
    at::Tensor indices_scratch = at::empty_like(arr, arr.options().dtype(torch::kInt64));
    at::Tensor tile_hist =
        at::empty({m, grid_n_hier, num_bins}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));

    const TritonJITFunction& tile_hist_kernel =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                        "compute_tile_hist_kernel");
    const TritonJITFunction& hierarchical_sweep_kernel =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                        "sweep_hierarchical_scatter");
    const int32_t grid_n_hier_i32 = static_cast<int32_t>(grid_n_hier);
    for (int64_t i = 0; i < n_passes; ++i) {
      tile_hist_kernel(raw_stream,
                       static_cast<unsigned int>(m),
                       static_cast<unsigned int>(grid_n_hier),
                       1,
                       4,
                       1,
                       arr_in,
                       tile_hist,
                       static_cast<int32_t>(i),
                       static_cast<int32_t>(i * k_bits),
                       m_i32,
                       n_i32,
                       grid_n_hier_i32,
                       TILE_N_HIER,
                       k_bits,
                       descending);
      at::Tensor pass_hist = tile_hist.sum({1}, false).to(torch::kInt32);
      at::Tensor pass_prefix = (at::cumsum(pass_hist, -1) - pass_hist).to(torch::kInt32);
      ex_cumsum_bins.select(1, i).copy_(pass_prefix);
      // The metadata is tiny compared with the payload.  Keep the scan in
      // int32 (N is bounded below 2**30) so no 64-bit prefix reaches scatter.
      at::Tensor tile_prefix = (at::cumsum(tile_hist, 1).to(torch::kInt32) - tile_hist).contiguous();
      hierarchical_sweep_kernel(raw_stream,
                                static_cast<unsigned int>(m),
                                static_cast<unsigned int>(grid_n_hier),
                                1,
                                4,
                                1,
                                arr_in,
                                indices_in,
                                arr_out,
                                indices_out,
                                ex_cumsum_bins,
                                tile_prefix,
                                static_cast<int32_t>(n_passes),
                                static_cast<int32_t>(i),
                                static_cast<int32_t>(i * k_bits),
                                m_i32,
                                n_i32,
                                grid_n_hier_i32,
                                TILE_N_HIER,
                                k_bits,
                                descending,
                                i == 0);
      if (i == 0) {
        arr_in = arr_out;
        arr_out = arr_scratch;
      } else if (i < n_passes - 1) {
        std::swap(arr_in, arr_out);
      } else {
        // The last pass writes the opposite ping-pong buffer; expose it
        // directly so no post-sort copy is needed.
        arr_in = arr_out;
      }
      std::swap(indices_in, indices_out);
    }
    return std::make_tuple(arr_in, indices_in);
  }

  // Keep pass zero read-only with respect to the user input.  The sweep
  // synthesizes its stable row offsets on pass zero, so this avoids both
  // clone(arr) and arange().broadcast_to(...).contiguous().
  at::Tensor arr_in = arr;
  at::Tensor arr_out = at::empty_like(arr);
  at::Tensor arr_scratch = at::empty_like(arr);

  const auto index_dtype = use_compact_indices ? torch::kInt32 : torch::kInt64;
  at::Tensor indices_out = at::empty_like(arr, arr.options().dtype(index_dtype));
  at::Tensor indices_scratch = at::empty_like(arr, arr.options().dtype(index_dtype));
  // Pass zero has a constexpr synthesize_indices flag and does not read this
  // pointer.  It must nevertheless be valid: the C++ JIT's nullopt pointer
  // ABI does not match Triton's removed-None pointer specialization.
  at::Tensor indices_in = indices_out;
  at::Tensor final_indices =
      use_compact_indices ? at::empty_like(arr, arr.options().dtype(torch::kInt64)) : at::Tensor();
#else
  at::Tensor arr_in = arr.clone();
  at::Tensor indices_in = at::arange(0, n, at::TensorOptions().dtype(torch::kInt64).device(arr.device()))
                              .broadcast_to(arr.sizes())
                              .contiguous();
  at::Tensor arr_out = at::empty_like(arr_in);
  at::Tensor indices_out = at::empty_like(indices_in);
#endif

  const int64_t TILE_R_SWEEP = 8;
  const int64_t TILE_N_SWEEP = 2048;
  int64_t grid_r_sweep = utils::cdiv(num_bins, TILE_R_SWEEP);
  int64_t grid_n_sweep = utils::cdiv(n, TILE_N_SWEEP);
#if defined(FLAGGEMS_USE_MUSA)
  const int32_t grid_n_sweep_i32 = static_cast<int32_t>(grid_n_sweep);
#endif
#if defined(FLAGGEMS_USE_GCU)
  int64_t total_tasks_sweep = m * grid_n_sweep;
  unsigned int grid_x_sweep = std::min(total_tasks_sweep, (int64_t)48);
#else
  unsigned int grid_x_sweep = m * grid_n_sweep;
#endif
  unsigned int grid_y_sweep = grid_r_sweep;

  at::Tensor status =
      at::empty({m, num_bins, grid_n_sweep}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));

  const TritonJITFunction& sweep_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                      "sweep");

  for (int64_t i = 0; i < n_passes; ++i) {
    int64_t bit_offset = i * k_bits;
    status.zero_();
#if defined(FLAGGEMS_USE_MUSA)
    const at::Tensor& pass_indices_out =
        (use_compact_indices && i == n_passes - 1) ? final_indices : indices_out;
#endif
    sweep_kernel(raw_stream,
                 grid_x_sweep,
                 grid_y_sweep,
                 1,
                 4,
                 1,
                 arr_in,
                 indices_in,
                 arr_out,
#if defined(FLAGGEMS_USE_MUSA)
                 pass_indices_out,
#else
                 indices_out,
#endif
                 ex_cumsum_bins,
                 status,
#if defined(FLAGGEMS_USE_MUSA)
                 static_cast<int32_t>(n_passes),
                 static_cast<int32_t>(i),
                 static_cast<int32_t>(bit_offset),
                 m_i32,
                 n_i32,
                 grid_n_sweep_i32,
#else
                 n_passes,
                 i,
                 bit_offset,
                 m,
                 n,
                 grid_n_sweep,
#endif
#if defined(FLAGGEMS_USE_GCU)
                 total_tasks_sweep,
#endif
                 TILE_N_SWEEP,
                 TILE_R_SWEEP,
                 k_bits,
#if defined(FLAGGEMS_USE_MUSA)
                 descending,
                 i == 0);
#else
                 descending);
#endif

#if defined(FLAGGEMS_USE_MUSA)
    if (i == 0) {
      arr_in = arr_out;
      arr_out = arr_scratch;
      indices_in = indices_out;
      std::swap(indices_out, indices_scratch);
    } else if (i < n_passes - 1) {
      at::Tensor recycled_arr = arr_in;
      arr_in = arr_out;
      arr_out = recycled_arr;
      at::Tensor recycled_indices = indices_in;
      indices_in = indices_out;
      indices_out = recycled_indices;
    } else {
      at::Tensor last_input = arr_in;
      arr_in = arr_out;
      arr_out = last_input;
    }
#else
    std::swap(arr_in, arr_out);
    std::swap(indices_in, indices_out);
#endif
  }

#if defined(FLAGGEMS_USE_MUSA)
  return std::make_tuple(arr_in, use_compact_indices ? final_indices : indices_out);
#else
  return std::make_tuple(arr_in, indices_in);
#endif
#endif  // FLAGGEMS_USE_MLU
}

std::tuple<at::Tensor, at::Tensor> sort_stable(const at::Tensor& inp,
                                               c10::optional<bool> stable,
                                               int64_t dim,
                                               bool descending) {
  if (inp.numel() == 0) {
    at::Tensor empty_out = at::empty_like(inp);
    at::Tensor empty_idx = at::empty_like(inp, at::TensorOptions().dtype(torch::kInt64));
    return std::make_tuple(empty_out, empty_idx);
  }
  int64_t ndim = inp.dim();
  int64_t original_dim = at::maybe_wrap_dim(dim, ndim);

  if (inp.size(original_dim) == 1) {
    return std::make_tuple(inp.clone(), at::zeros_like(inp, at::TensorOptions().dtype(torch::kInt64)));
  }

  at::Tensor contiguous_inp = inp;
  if (original_dim != ndim - 1) {
    contiguous_inp = inp.movedim(original_dim, -1).contiguous();
  } else {
    contiguous_inp = inp.contiguous();
  }

#if defined(FLAGGEMS_USE_MUSA)
  const auto dtype = contiguous_inp.scalar_type();
  const int64_t n = contiguous_inp.size(-1);
  const int64_t m = contiguous_inp.numel() / n;
  const int64_t local_limit = dtype == torch::kInt64 ? 512 : 1024;
  if (dtype != torch::kBool && !c10::isFloatingType(dtype) && n <= local_limit) {
    at::Tensor out = at::empty_like(contiguous_inp);
    at::Tensor out_index = at::empty_like(contiguous_inp, contiguous_inp.options().dtype(torch::kInt64));
    const TritonJITFunction& local_sort_kernel =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / SORT_KERNEL_PATH),
                                        "sort_kernel");

    c10::DeviceGuard guard(contiguous_inp.device());
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);
    local_sort_kernel(raw_stream,
                      m,
                      1,
                      1,
                      4,
                      1,
                      contiguous_inp,
                      out,
                      out_index,
                      n,
                      utils::next_power_of_2(n),
                      descending,
                      false);

    if (original_dim != ndim - 1) {
      out = out.movedim(-1, original_dim);
      out_index = out_index.movedim(-1, original_dim);
    }
    return std::make_tuple(out, out_index);
  }
#endif

  int64_t k_bits = (contiguous_inp.scalar_type() == torch::kBool) ? 1 : 4;
  auto [out, out_index] = radix_sort(contiguous_inp, k_bits, descending);

  if (original_dim != ndim - 1) {
    out = out.movedim(-1, original_dim);
    out_index = out_index.movedim(-1, original_dim);
  }

  return std::make_tuple(out, out_index);
}

std::tuple<at::Tensor, at::Tensor> sort(const at::Tensor& inp, int64_t dim, bool descending) {
  return sort_stable(inp, false, dim, descending);
}

}  // namespace flag_gems
