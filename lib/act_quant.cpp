#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <iostream>
#include "ATen/WrapDimUtils.h"
#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

std::vector<at::Tensor> act_quant_triton(const at::Tensor& x,
                                         int block_size,
                                         std::optional<std::string> scale_fmt) {
  TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
  TORCH_CHECK(x.size(-1) % block_size == 0, "Last dimension size must be divisible by block_size");

  int N = x.size(-1);
  auto x_2d = x.view({-1, N});
  int M = x_2d.size(0);

  int BLOCK_M = 32;
  int BLOCK_N = block_size;
  int m_blocks = utils::cdiv(M, BLOCK_M);
  int n_blocks = N / BLOCK_N;

  at::Tensor y = torch::empty_like(x, torch::kFloat8_e4m3fn);
  // at::Tensor y = at::empty(x.sizes(), at::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(x.device()));
  //   at::Tensor y = torch::empty_like(x, torch::kFloat32);

  std::vector<int64_t> s_shape;
  s_shape.insert(s_shape.end(), x.sizes().begin(), x.sizes().end() - 1);
  s_shape.push_back(n_blocks);
  at::Tensor s = x.new_empty(s_shape, torch::kFloat32);
  at::Tensor y_view = y.view({-1, N});
  at::Tensor s_view = s.view({-1, n_blocks});

  // TORCH_CHECK(reinterpret_cast<uintptr_t>(y_view.data_ptr()) % 16 == 0, "act_quant_backup1.cpp must be
  // 16-byte aligned for efficient vectorized store");

  const TritonJITFunction& kernel =
      TritonJITFunction::get_instance(std::string(utils::get_flag_gems_src_path() / "ops" / "act_quant.py"),
                                      "act_quant_triton_kernel");
  c10::DeviceGuard guard(y.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  // void* p_item = y.data_ptr();
  // std::cout << "y is 16-byte aligned: " << (reinterpret_cast<std::uintptr_t>(p_item) % 16 == 0) <<
  // std::endl; auto y_ptr = reinterpret_cast<uintptr_t>(y_view.data_ptr()); auto x_ptr =
  // reinterpret_cast<uintptr_t>(x_2d.data_ptr()); auto s_ptr =
  // reinterpret_cast<uintptr_t>(s_view.data_ptr()); if (y_ptr % 16 != 0) {
  //     fprintf(stderr, "WARNING: y_view data_ptr=%p NOT 16-byte aligned (mod16=%lu), shape=[%ld,%ld]\n",
  //         y_view.data_ptr(), y_ptr % 16, M, N);
  // }

  // if (x.sizes().size() == 3 && x.sizes()[0] == 1 && x.sizes()[1] == 1 && x.sizes()[2] == 448) {
  //   kernel(/* CUstream = */ raw_stream,
  //          /* grid_x = */ m_blocks,
  //          /* grid_y = */ n_blocks,
  //          /* grid_z = */ 1,
  //          /* num_warps = */ 4,
  //          /* num_stages = */ 3,
  //          x_2d,
  //          y_view,
  //          s_view,
  //          M,
  //          N,
  //          (int)x_2d.stride(0),
  //          (int)y_view.stride(0),
  //          (int)s_view.stride(0),
  //          BLOCK_M,
  //          BLOCK_N,
  //          scale_fmt.has_value());
  // } else {
  //   // kernel(/* CUstream = */ raw_stream,
  //   //        /* grid_x = */ m_blocks,
  //   //        /* grid_y = */ n_blocks,
  //   //        /* grid_z = */ 1,
  //   //        /* num_warps = */ 4,
  //   //        /* num_stages = */ 3,
  //   //        x_2d,
  //   //        y_view,
  //   //        s_view,
  //   //        M,
  //   //        N,
  //   //        (int)x_2d.stride(0),
  //   //        (int)y_view.stride(0),
  //   //        (int)s_view.stride(0),
  //   //        BLOCK_M,
  //   //        BLOCK_N,
  //   //        scale_fmt.has_value());
  // }
  //   return {y.to(torch::kFloat8_e4m3fn), s};

  kernel(/* CUstream = */ raw_stream,
         /* grid_x = */ m_blocks,
         /* grid_y = */ n_blocks,
         /* grid_z = */ 1,
         /* num_warps = */ 4,
         /* num_stages = */ 3,
         x_2d,
         y_view,
         s_view,
         M,
         N,
         (int)x_2d.stride(0),
         (int)y_view.stride(0),
         (int)s_view.stride(0),
         BLOCK_M,
         BLOCK_N,
         scale_fmt.has_value());

  return {y, s};
}

}  // namespace flag_gems
