#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "ATen/WrapDimUtils.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

int64_t get_num_bits(const at::ScalarType& dtype) {
  if (dtype == torch::kBool) {
    return 1;
  }
  return c10::elementSize(dtype) * 8;
}

std::tuple<at::Tensor, at::Tensor> radix_sort(const at::Tensor& arr, int64_t k_bits, bool descending) {
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
  unsigned int grid_x_hist = m * grid_n_hist;

  const TritonJITFunction& hist_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sort.py"),
                                      "compute_global_hist_kernel");

  c10::DeviceGuard guard(arr.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

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
  unsigned int grid_x_sweep = m * grid_n_sweep;
  unsigned int grid_y_sweep = grid_r_sweep;

  at::Tensor status =
      at::empty({m, num_bins, grid_n_sweep}, at::TensorOptions().device(arr.device()).dtype(torch::kInt32));

  const TritonJITFunction& sweep_kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sort.py"),
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
                 TILE_N_SWEEP,
                 TILE_R_SWEEP,
                 k_bits,
                 descending);

    std::swap(arr_in, arr_out);
    std::swap(indices_in, indices_out);
  }

  return std::make_tuple(arr_in, indices_in);
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
