#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

#include <filesystem>
#include "ATen/WrapDimUtils.h"
#include "ATen/native/ReduceOpsUtils.h"
#include "c10/util/DimVector.h"

namespace flag_gems {
using namespace triton_jit;
// sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
at::Tensor sum(const at::Tensor &self, ::std::optional<at::ScalarType> dtype) {
  TORCH_CHECK(self.is_contiguous(), "Input tensor must be contiguous");
  int64_t M = self.numel();
  int64_t block_size = utils::next_power_of_2(static_cast<int>(std::ceil(std::sqrt(M))));
  int64_t mid_size = utils::cdiv(M, block_size);
  int64_t block_mid = utils::next_power_of_2(mid_size);
  at::Tensor mid = torch::empty({mid_size}, self.options());
  at::Tensor out = torch::empty({}, self.options());
  const TritonJITFunction &sum_kernel_1 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                      "sum_kernel_1");
  const TritonJITFunction &sum_kernel_2 =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                      "sum_kernel_2");
  const int num_warps = 8;
  const int num_stages = 2;
  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  sum_kernel_1(raw_stream, mid_size, 1, 1, num_warps, num_stages, self, mid, M, block_size);
  sum_kernel_2(raw_stream, 1, 1, 1, num_warps, num_stages, mid, out, mid_size, block_mid);
  return out;
}

// signature
// sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType?
// dtype=None) -> Tensor
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   ::std::optional<at::ScalarType> dtype) {
  at::TensorOptions out_options = self.options();
  at::ScalarType out_dtype;
  if (dtype.has_value()) {
    out_dtype = dtype.value();
  } else {
    out_dtype = self.dtype().toScalarType();
    ;
    if (out_dtype == torch::kBool) {
      out_dtype = torch::kInt64;
    }
  }
  out_options = out_options.dtype(out_dtype);
  at::DimVector dims_ = at::native::make_dim_vector(dim, self.dim());
  at::maybe_wrap_dims(dims_, self.dim());
  at::DimVector shape = at::meta::get_reduction_shape(self, dims_, keepdim, false);
  at::Tensor out = at::empty(at::IntArrayRef(shape), out_options);
  out = out.contiguous();

  if (!dim.has_value() || dim->empty()) {
    if (!keepdim) {
      return flag_gems::sum(self, std::optional<at::ScalarType> {});
    } else {
      at::Tensor result = flag_gems::sum(self, dtype);
      return result.reshape(std::vector<int64_t>(self.dim(), 1));
    }
  }
  int64_t tile_m = 4;
  int64_t tile_n = 512;
  int64_t tile_k = 4;
  const int num_warps = 8;
  const int num_stages = 2;
  c10::DeviceGuard guard(self.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  at::Tensor self_contiguous = self.contiguous();
  if (dims_.size() == 1) {
    int64_t selected_dim = dims_[0];
    // M, N, K in python sum_dim_comm
    auto [non_reduction_size, reduction_size, remain_size] = utils::parse_reduction_axes(self, selected_dim);
    bool ONE_TILE_PER_CTA = (tile_n >= reduction_size);
    if (remain_size > 1) {
      const TritonJITFunction &f =
          TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                          "sum_dim_kernel_non_inner");
      f(raw_stream,
        non_reduction_size,
        utils::cdiv(remain_size, tile_k),
        1,
        num_warps,
        num_stages,
        out,
        self_contiguous,
        non_reduction_size,
        reduction_size,
        remain_size,
        tile_n,
        tile_k,
        ONE_TILE_PER_CTA);
    } else {
      auto [non_reduction_size, reduction_size, remain_size] =
          utils::parse_reduction_axes(self, selected_dim);
      const TritonJITFunction &f =
          TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                          "sum_dim_kernel_inner");
      f(raw_stream,
        non_reduction_size,
        1,
        1,
        num_warps,
        num_stages,
        out,
        self_contiguous,
        non_reduction_size,
        reduction_size,
        tile_n,
        ONE_TILE_PER_CTA);
    }
    return out;
  } else {
    auto [permuted_self, non_reduction_size, reduction_size] =
        utils::permute_reduction_axes_right(self, dims_);
    const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;
    permuted_self = permuted_self.contiguous();
    /* signature to remind yourself
    def sum_kernel(
      in_ptr,
      out_ptr,
      M,
      N,
      BLOCK_M: tl.constexpr,
      BLOCK_N: tl.constexpr,
      STAGE: tl.constexpr,
    ):
    */
    const TritonJITFunction &f =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "sum.py"),
                                        "sum_dim_kernel");
    c10::DeviceGuard guard(out.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    f(raw_stream,
      num_blocks,
      1,
      1,
      num_warps,
      num_stages,
      permuted_self,
      out,
      non_reduction_size,
      reduction_size,
      tile_m,
      tile_n);
    return out;
  }
}

}  // namespace flag_gems
