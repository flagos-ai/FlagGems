#include "flag_gems/operators.h"
#include "pointwise_runtime.h"

namespace flag_gems {

// fill.Scalar(Tensor self, Scalar value) -> Tensor
at::Tensor fill_scalar(const at::Tensor& input, const c10::Scalar& value) {
  double value_val = value.toDouble();
  return pointwise_dynamic::fill_scalar_func(input, value_val);
}

// fill.Tensor(Tensor self, Tensor value) -> Tensor
at::Tensor fill_tensor(const at::Tensor& input, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor only supports 0-dim value tensor");
  return pointwise_dynamic::fill_tensor_func(input, value);
}

// fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
at::Tensor& fill_scalar_(at::Tensor& input, const c10::Scalar& value) {
  double value_val = value.toDouble();
  pointwise_dynamic::fill_scalar_func_out(input, input, value_val);
  return input;
}

// fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
at::Tensor& fill_tensor_(at::Tensor& input, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor_ only supports 0-dim value tensor");
  pointwise_dynamic::fill_tensor_func_out(input, value, input);
  return input;
}

}  // namespace flag_gems

// NOTE:
// Deprecated and scheduled for removal in v4.4.
/***
at::Tensor fill_scalar(const at::Tensor& input, const c10::Scalar& value) {
  at::Tensor out = at::empty_like(input);
  int64_t numel = out.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel = get_fill_scalar_kernel();

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);
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

  const TritonJITFunction& fill_kernel = get_fill_tensor_kernel();

  c10::DeviceGuard guard(out.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, out, value, numel, BLOCK_SIZE);

  return out;
}

at::Tensor& fill_scalar_(at::Tensor& input, const c10::Scalar& value) {
  int64_t numel = input.numel();
  if (numel == 0) return input;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel = get_fill_scalar_kernel();

  c10::DeviceGuard guard(input.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, input, value, numel, BLOCK_SIZE);

  return input;
}

at::Tensor& fill_tensor_(at::Tensor& input, const at::Tensor& value) {
  TORCH_CHECK(value.dim() == 0, "fill_tensor_ only supports 0-dim value tensor");
  int64_t numel = input.numel();
  if (numel == 0) return input;

  constexpr int BLOCK_SIZE = 1024;
  unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const TritonJITFunction& fill_kernel = get_fill_tensor_kernel();

  c10::DeviceGuard guard(input.device());
  backend::StreamType stream = backend::getCurrentStream();
  backend::RawStreamType raw_stream = backend::getRawStream(stream);

  fill_kernel(raw_stream, grid_x, 1, 1, 4, 0, input, value, numel, BLOCK_SIZE);

  return input;
}
***/
