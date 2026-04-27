#include "flag_gems/operators.h"
#include "pointwise_runtime.h"

namespace flag_gems {

// true_div: Tensor / Tensor
at::Tensor true_div(const at::Tensor &a, const at::Tensor &b) {
  if (a.dim() == 0 && b.dim() > 0) {
    return pointwise_dynamic::true_div_func_scalar_tensor(b, a.item<double>());
  }
  if (b.dim() == 0) {
    return pointwise_dynamic::true_div_func_tensor_scalar(a, b.item<double>());
  }
  return pointwise_dynamic::true_div_func(a, b);
}

// true_div_: in-place
at::Tensor true_div_(at::Tensor &a, const at::Tensor &b) {
  if (b.dim() == 0) {
    pointwise_dynamic::true_div_func_tensor_scalar_out(a, a, b.item<double>());
  } else {
    pointwise_dynamic::true_div_func_out(a, b, a);
  }
  return a;
}

// trunc_div: trunc(a / b)
at::Tensor trunc_div(const at::Tensor &a, const at::Tensor &b) {
  if (a.dim() == 0 && b.dim() > 0) {
    return pointwise_dynamic::trunc_div_func_scalar_tensor(b, a.item<double>());
  }
  if (b.dim() == 0) {
    return pointwise_dynamic::trunc_div_func_tensor_scalar(a, b.item<double>());
  }
  return pointwise_dynamic::trunc_div_func(a, b);
}

// trunc_div_: in-place
at::Tensor trunc_div_(at::Tensor &a, const at::Tensor &b) {
  if (b.dim() == 0) {
    pointwise_dynamic::trunc_div_func_tensor_scalar_out(a, a, b.item<double>());
  } else {
    pointwise_dynamic::trunc_div_func_out(a, b, a);
  }
  return a;
}

// floor_div: floor(a / b)
at::Tensor floor_div(const at::Tensor &a, const at::Tensor &b) {
  if (a.dim() == 0 && b.dim() > 0) {
    return pointwise_dynamic::floor_div_func_scalar_tensor(b, a.item<double>());
  }
  if (b.dim() == 0) {
    return pointwise_dynamic::floor_div_func_tensor_scalar(a, b.item<double>());
  }
  return pointwise_dynamic::floor_div_func(a, b);
}

// floor_div_: in-place
at::Tensor floor_div_(at::Tensor &a, const at::Tensor &b) {
  if (b.dim() == 0) {
    pointwise_dynamic::floor_div_func_tensor_scalar_out(a, a, b.item<double>());
  } else {
    pointwise_dynamic::floor_div_func_out(a, b, a);
  }
  return a;
}

// div_mode: dispatch based on rounding_mode
at::Tensor div_mode(const at::Tensor &a,
                    const at::Tensor &b,
                    const c10::optional<std::string> &rounding_mode) {
  if (rounding_mode == "floor") {
    return floor_div(a, b);
  } else if (rounding_mode == "trunc") {
    return trunc_div(a, b);
  } else if (!rounding_mode.has_value() || rounding_mode.value() == "none") {
    return true_div(a, b);
  }
  TORCH_CHECK(false, "div_mode: rounding_mode must be 'floor', 'trunc', or empty.");
}

// div_mode_: in-place dispatch
at::Tensor div_mode_(at::Tensor &a, const at::Tensor &b, const c10::optional<std::string> &rounding_mode) {
  if (rounding_mode == "floor") {
    return floor_div_(a, b);
  } else if (rounding_mode == "trunc") {
    return trunc_div_(a, b);
  } else if (!rounding_mode.has_value() || rounding_mode.value() == "none") {
    return true_div_(a, b);
  }
  TORCH_CHECK(false, "div_mode_: rounding_mode must be 'floor', 'trunc', or empty.");
}

// remainder: tensor % tensor
at::Tensor remainder_tt(const at::Tensor &a, const at::Tensor &b) {
  return pointwise_dynamic::rem_tt(a, b);
}

at::Tensor remainder_ts(const at::Tensor &a, double b_scalar) {
  return pointwise_dynamic::rem_ts(a, b_scalar);
}

at::Tensor remainder_st(double a_scalar, const at::Tensor &b) {
  return pointwise_dynamic::rem_st(b, a_scalar);
}

at::Tensor remainder(const at::Tensor &a, const at::Tensor &b) {
  if (a.dim() == 0 && b.dim() == 0) {
    double a_val = a.item<double>();
    double b_val = b.item<double>();
    double r = std::fmod(a_val, b_val);
    if (r != 0.0 && ((r < 0.0) != (b_val < 0.0))) {
      r += b_val;
    }
    return torch::tensor(r, a.options());
  }
  if (a.dim() > 0 && b.dim() > 0) {
    return remainder_tt(a, b);
  }
  if (a.dim() > 0 && b.dim() == 0) {
    return remainder_ts(a, b.item<double>());
  }
  // Scalar % Tensor
  return remainder_st(a.item<double>(), b);
}

at::Tensor remainder_(at::Tensor &a, const at::Tensor &b) {
  if (b.dim() == 0) {
    pointwise_dynamic::rem_ts_out(a, a, b.item<double>());
  } else {
    pointwise_dynamic::rem_tt_out(a, b, a);
  }
  return a;
}

}  // namespace flag_gems
