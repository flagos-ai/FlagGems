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

namespace {
std::tuple<at::Tensor, int64_t, int64_t> permute_reduction_axes_right(
    const at::Tensor &tensor, at::OptionalIntArrayRef reduction_axes_opt) {
  int64_t dim = tensor.dim();
  c10::DimVector reduction_axes;

  if (reduction_axes_opt.has_value()) {
    reduction_axes = reduction_axes_opt.value().vec();
  }

  std::unordered_set<int64_t> reduction_set(reduction_axes.begin(), reduction_axes.end());

  c10::DimVector left_axes, right_axes;
  int64_t non_reduction_size = 1, reduction_size = 1;

  for (int64_t i = 0; i < dim; ++i) {
    if (reduction_set.count(i)) {
      right_axes.push_back(i);
      reduction_size *= tensor.size(i);
    } else {
      left_axes.push_back(i);
      non_reduction_size *= tensor.size(i);
    }
  }

  // Concatenate left and right axes to form the new permutation order
  c10::DimVector permute_order = left_axes;
  permute_order.insert(permute_order.end(), right_axes.begin(), right_axes.end());

  return {tensor.permute(permute_order), non_reduction_size, reduction_size};
}
}  // anonymous namespace

namespace flag_gems {
using namespace triton_jit;
// signature
// sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType?
// dtype=None) -> Tensor
at::Tensor sum_dim(const at::Tensor &self,
                   at::OptionalIntArrayRef dim,
                   bool keepdim,
                   ::std::optional<at::ScalarType> dtype) {
  at::DimVector dims_ = at::native::make_dim_vector(dim, self.dim());
  at::maybe_wrap_dims(dims_, self.dim());
  at::DimVector shape = at::meta::get_reduction_shape(self, dims_, keepdim, false);
  c10::ScalarType out_dtype = at::native::get_dtype_from_self(self, dtype, true);
  at::Tensor out = at::empty(shape, self.options());

  auto [permuted_self, non_reduction_size, reduction_size] = permute_reduction_axes_right(self, dims_);
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
      TritonJITFunction::getInstance(std::string(utils::get_triton_src_path() / "sum.py"), "sum_kernel");

  // add utility to build this automatically
  int64_t tile_m = 4;
  int64_t tile_n = 512;
  const int num_warps = 8;
  const int num_stages = 2;
  const unsigned int num_blocks = (non_reduction_size + tile_m - 1) / tile_m;

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(stream,
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
    tile_n,
    num_stages);
  return out;
}

}  // namespace flag_gems
