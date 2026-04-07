#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include "flag_gems/backend_utils.h"
#include "triton_jit/triton_jit_function.h"
#include <iostream>

namespace flag_gems {
using namespace triton_jit;

at::Tensor fp8_matmul(const at::Tensor& a,
                      const at::Tensor& a_s,
                      const at::Tensor& b,
                      const at::Tensor& b_s) {
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "a and b must be contiguous");
  TORCH_CHECK(a_s.is_contiguous() && b_s.is_contiguous(),
              "a_s and b_s must be contiguous");

  int64_t K = a.size(-1);
  int64_t M = a.numel() / K;
  int64_t N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "a and b must have the same K dimension");

  auto out_shape = a.sizes().vec();
  out_shape.back() = N;

  at::Tensor a_2d = a.view({M, K});
  at::Tensor a_s_2d = a_s.view({M, -1});
  at::Tensor C = at::empty({M, N}, a.options().dtype(at::kBFloat16));

  const int BLOCK_M = 64;
  const int BLOCK_N = 64;
  const int BLOCK_K = 128;
  const int GROUP_SIZE_M = 4;
  const int GROUP_K = 128;
  const int num_warps = 4;
  const int num_stages = 3;

  int64_t grid_x = utils::cdiv(static_cast<int>(M), BLOCK_M) *
                   utils::cdiv(static_cast<int>(N), BLOCK_N);

  const TritonJITFunction& kernel =
      TritonJITFunction::get_instance(
          std::string(utils::get_flag_gems_src_path() / "ops" / "fp8_matmul.py"),
          "_fp8_matmul_kernel");

  c10::DeviceGuard guard(C.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  kernel(raw_stream,
         /* grid_x = */ grid_x,
         /* grid_y = */ 1,
         /* grid_z = */ 1,
         /* num_warps = */ num_warps,
         /* num_stages = */ num_stages,
         a_2d, b, C,
         a_s_2d, b_s,
         static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
         static_cast<int>(a_2d.stride(0)), static_cast<int>(a_2d.stride(1)),
         static_cast<int>(b.stride(0)), static_cast<int>(b.stride(1)),
         static_cast<int>(C.stride(0)), static_cast<int>(C.stride(1)),
         static_cast<int>(a_s_2d.stride(0)), static_cast<int>(a_s_2d.stride(1)),
         static_cast<int>(b_s.stride(0)), static_cast<int>(b_s.stride(1)),
         GROUP_K,
         BLOCK_M, BLOCK_N, BLOCK_K,
         GROUP_SIZE_M);

  return C.view(out_shape);
}

at::Tensor fp8_matmul_noop(const at::Tensor& a,
                           const at::Tensor& a_s,
                           const at::Tensor& b,
                           const at::Tensor& b_s) {
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
              "a and b must be contiguous");
  TORCH_CHECK(a_s.is_contiguous() && b_s.is_contiguous(),
              "a_s and b_s must be contiguous");

  int64_t K = a.size(-1);
  int64_t M = a.numel() / K;
  int64_t N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "a and b must have the same K dimension");

  auto out_shape = a.sizes().vec();
  out_shape.back() = N;

  at::Tensor a_2d = a.view({M, K});
  at::Tensor a_s_2d = a_s.view({M, -1});
  at::Tensor C = at::empty({M, N}, a.options().dtype(at::kBFloat16));

  return C.view(out_shape);
}

}  // namespace flag_gems
