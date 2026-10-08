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

  auto dtype = arr.scalar_type();
  int64_t num_bits = get_num_bits(dtype);

  const int64_t TILE_N_HIST = 1024;
  const int64_t TILES_N_PER_CTA_HIST = 8;
  const int64_t CTA_TILE_N_HIST = TILES_N_PER_CTA_HIST * TILE_N_HIST;

  const int64_t num_bins = 1 << k_bits;
  const int64_t n_passes = utils::cdiv(num_bits, k_bits);
  const int64_t TILE_R_HIST = 16;

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
              n_passes,
              m,
              n,
#if defined(FLAGGEMS_USE_GCU)
              grid_n_hist,
#endif
              TILES_N_PER_CTA_HIST,
              TILE_N_HIST,
              TILE_R_HIST,
              k_bits,
              descending);

  at::Tensor ex_cumsum_bins = at::cumsum(global_hist, -1) - global_hist;
  ex_cumsum_bins = ex_cumsum_bins.to(torch::kInt32);

  at::Tensor arr_in = arr.clone();
  at::Tensor indices_in = at::arange(0, n, at::TensorOptions().dtype(torch::kInt64).device(arr.device()))
                              .broadcast_to(arr.sizes())
                              .contiguous();
  at::Tensor arr_out = at::empty_like(arr_in);
  at::Tensor indices_out = at::empty_like(indices_in);

  const int64_t TILE_R_SWEEP = 8;
  const int64_t TILE_N_SWEEP = 2048;
  int64_t grid_r_sweep = utils::cdiv(num_bins, TILE_R_SWEEP);
  int64_t grid_n_sweep = utils::cdiv(n, TILE_N_SWEEP);
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
    sweep_kernel(raw_stream,
                 grid_x_sweep,
                 grid_y_sweep,
                 1,
                 4,
                 1,
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
#if defined(FLAGGEMS_USE_GCU)
                 total_tasks_sweep,
#endif
                 TILE_N_SWEEP,
                 TILE_R_SWEEP,
                 k_bits,
                 descending);

    std::swap(arr_in, arr_out);
    std::swap(indices_in, indices_out);
  }

  return std::make_tuple(arr_in, indices_in);
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
