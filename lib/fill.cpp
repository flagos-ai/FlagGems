#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;

at::Tensor fill_scalar(const at::Tensor& input, const c10::Scalar& value) {
  at::Tensor out = at::empty_like(input);
  int64_t numel = out.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "fill.py").string(),
                                      "fill_scalar_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, out, value, numel, BLOCK_SIZE);

  return out;
}

at::Tensor fill_tensor(const at::Tensor& input, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor only supports 0-dim value tensor");
  at::Tensor out = at::empty_like(input);
  int64_t numel = out.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "fill.py").string(),
                                      "fill_tensor_kernel");

  c10::DeviceGuard guard(out.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, out, value, numel, BLOCK_SIZE);

  return out;
}

at::Tensor& fill_scalar_(at::Tensor& input, const c10::Scalar& value) {
  int64_t numel = input.numel();
  if (numel == 0) return input;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "fill.py").string(),
                                      "fill_scalar_kernel");

  c10::DeviceGuard guard(input.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, input, value, numel, BLOCK_SIZE);
  return input;
}

at::Tensor& fill_tensor_(at::Tensor& input, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor_ only supports 0-dim value tensor");
  int64_t numel = input.numel();
  if (numel == 0) return input;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "fill.py").string(),
                                      "fill_tensor_kernel");

  c10::DeviceGuard guard(input.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, input, value, numel, BLOCK_SIZE);
  return input;
}

}  // namespace flag_gems
