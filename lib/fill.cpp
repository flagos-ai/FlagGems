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
