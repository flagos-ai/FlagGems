#include "flag_gems/operators.h"
#include "pointwise_runtime.h"

namespace flag_gems {

// add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
at::Tensor add_tensor(const at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  double alpha_val = alpha.toDouble();
  return pointwise_dynamic::add_func(self, other, alpha_val);
}

// add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
at::Tensor add_scalar(const at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
  double other_val = other.toDouble();
  double alpha_val = alpha.toDouble();
  return pointwise_dynamic::add_func_tensor_scalar(self, other_val, alpha_val);
}

// add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
at::Tensor &add_tensor_inplace(at::Tensor &self, const at::Tensor &other, const at::Scalar &alpha) {
  double alpha_val = alpha.toDouble();
  pointwise_dynamic::add_func_out(self, other, self, alpha_val);
  return self;
}

// add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
at::Tensor &add_scalar_inplace(at::Tensor &self, const at::Scalar &other, const at::Scalar &alpha) {
  double other_val = other.toDouble();
  double alpha_val = alpha.toDouble();
  pointwise_dynamic::add_func_tensor_scalar_out(self, self, other_val, alpha_val);
  return self;
}

}  // namespace flag_gems
