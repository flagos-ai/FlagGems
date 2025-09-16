#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "ATen/WrapDimUtils.h"
#include "c10/cuda/CUDAStream.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor argmax(const at::Tensor &self, std::optional<int64_t> dim, bool keepdim) {
  if (!dim.has_value()) {
    int64_t M = self.numel();

    int64_t block_size = utils::next_power_of_2(static_cast<int64_t>(std::ceil(std::sqrt(M))));
    int64_t mid_size = (M + block_size - 1) / block_size;
    int64_t block_mid = utils::next_power_of_2(mid_size);

    at::Tensor mid_value = at::empty({mid_size}, self.options());
    at::Tensor mid_index = at::empty({mid_size}, self.options().dtype(at::kLong));

    at::Tensor out;
    if (keepdim) {
      const auto shape = std::vector<int64_t>(self.dim(), 1);
      out = at::empty(shape, self.options().dtype(at::kLong));
    } else {
      out = at::empty({}, self.options().dtype(at::kLong));
    }

    const TritonJITFunction &f1 =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "argmax.py"),
                                        "argmax_kernel_1");
    const TritonJITFunction &f2 =
        TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "argmax.py"),
                                        "argmax_kernel_2");

    c10::DeviceGuard guard(self.device());
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
    CUstream raw_stream = static_cast<CUstream>(stream.stream());
    f1(raw_stream,
       mid_size,
       1,
       1,
       /* num_warps = */ 4,
       /* num_stages = */ 2,
       self,
       mid_value,
       mid_index,
       M,
       block_size);

    f2(raw_stream,
       1,
       1,
       1,
       /* num_warps = */ 4,
       /* num_stages = */ 2,
       mid_value,
       mid_index,
       out,
       mid_size,
       block_mid);

    return out;
  }

  int64_t dim_val = dim.value();
  dim_val = at::maybe_wrap_dim(dim_val, self.dim());

  const auto &shape = self.sizes();
  int64_t N = shape[dim_val];
  int64_t M = 1;
  for (int64_t i = 0; i < dim_val; ++i) {
    M *= shape[i];
  }
  int64_t K = self.numel() / (M * N);

  at::DimVector out_shape;
  if (keepdim) {
    out_shape = shape.vec();
    out_shape[dim_val] = 1;
  } else {
    out_shape.reserve(shape.size() - 1);
    for (int64_t i = 0; i < shape.size(); ++i) {
      if (i != dim_val) {
        out_shape.push_back(shape[i]);
      }
    }
  }

  at::Tensor out = at::empty(out_shape, self.options().dtype(at::kLong));
  at::Tensor contiguous_self = self.contiguous();

  const TritonJITFunction &f =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "argmax.py"),
                                      "argmax_kernel");

  int64_t tile_m = 32;
  int64_t tile_n = 512;
  const int num_warps = 4;
  const int num_stages = 2;
  const unsigned int grid_x = (M + tile_m - 1) / tile_m;
  const unsigned int grid_y = K;

  c10::DeviceGuard guard(self.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(raw_stream, grid_x, grid_y, 1, num_warps, num_stages, contiguous_self, out, M, N, K, tile_m, tile_n);

  return out;
}

}  // namespace flag_gems
