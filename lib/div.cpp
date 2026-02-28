#include "flag_gems/operators.h"
#include "flag_gems/utils.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/util/TypeTraits.h>
#include <iostream>
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {
using namespace triton_jit;
at::Tensor full_like_cpp_scalar(const at::Tensor& x,
                                double fill_value,
                                c10::optional<c10::ScalarType> dtype_opt = c10::nullopt,
                                c10::optional<c10::Device> device_opt = c10::nullopt) {
  TORCH_CHECK(x.is_cuda(), "This example only supports CUDA tensors.");

  c10::ScalarType dtype = dtype_opt.value_or(x.scalar_type());
  c10::Device device = device_opt.value_or(x.device());

  c10::TensorOptions options = c10::TensorOptions().dtype(dtype).device(device);

  at::Tensor out = at::empty_like(x, options);

  if (out.numel() == 0) {
    return out;
  }

  static const TritonJITFunction& full_kernel_func =
      TritonJITFunction::get_instance(std::string(utils::get_triton_src_path() / "flag_gems/ops/full.py"),
                                      "full_func_scalar");

  c10::DeviceGuard guard(device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  constexpr int BLOCK_SIZE = 1024;
  int64_t num_elements = out.numel();
  unsigned int grid_x = static_cast<unsigned int>((num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);
  unsigned int grid_y = 1;
  unsigned int grid_z = 1;

  full_kernel_func(raw_stream, grid_x, grid_y, grid_z, 4, 2, out, fill_value);

  return out;
}

at::Tensor true_div(const at::Tensor& a, const at::Tensor& b) {
  // Case 1: Scalar / Scalar → fallback
  if (a.dim() == 0 && b.dim() == 0) {
    double a_val = a.item<double>();
    double b_val = b.item<double>();
    return torch::tensor(a_val / b_val, a.options());
  }

  // Case 2: Scalar / Tensor
  if (a.dim() == 0 && b.dim() > 0) {
    at::Tensor a = full_like_cpp_scalar(b, a.item<double>());
    return true_div(a, b);
  }

  // Case 3: Tensor / Scalar
  if (a.dim() > 0 && b.dim() == 0) {
    at::Tensor b = full_like_cpp_scalar(a, b.item<double>());
    return true_div(a, b);
  }

  // Case 4: Tensor / Tensor
  auto res = torch::broadcast_tensors({a, b});
  at::Tensor an = res[0].contiguous();
  at::Tensor bn = res[1].contiguous();

  at::ScalarType out_dtype = at::promote_types(an.scalar_type(), bn.scalar_type());
  at::Tensor out = at::empty(an.sizes(), at::TensorOptions().dtype(out_dtype).device(an.device()));

  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), "true_div_kernel");

  int64_t tile_size = 1024;
  int64_t n = out.numel();
  int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(out.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, an, bn, out, n, tile_size);

  return out;
}

at::Tensor true_div_(at::Tensor& a_, const at::Tensor& b_) {
  TORCH_CHECK(a_.device().is_cuda(), "true_div_: only CUDA tensors supported");
  TORCH_CHECK(a_.is_contiguous(), "true_div_: input a_ must be contiguous for in-place op");

  //  Tensor // Scalar
  if (a_.dim() > 0 && b_.dim() == 0) {
    at::Tensor b_expand = full_like_cpp_scalar(a_, b_.item<double>());
    return true_div_(a_, b_expand);
  }

  // Tensor // Tensor
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a_broadcast = res[0].contiguous();
  at::Tensor b_broadcast = res[1].contiguous();
  TORCH_CHECK(a_.numel() == a_broadcast.numel(), "true_div_: broadcasted shape mismatch");

  int64_t num = a_broadcast.numel();
  at::Tensor out = at::empty_like(a_broadcast);

  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), "true_div_kernel_");

  int64_t tile_size = 1024;
  int num_blocks = (num + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(a_.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, a_broadcast, b_broadcast, out, num, tile_size);

  return a_.resize_as_(out).copy_(out);
}

at::Tensor trunc_div(const at::Tensor& a, const at::Tensor& b) {
  // Scalar / Scalar fallback
  if (a.dim() == 0 && b.dim() == 0) {
    double a_val = a.item<double>();
    double b_val = b.item<double>();
    return torch::tensor(std::trunc(a_val / b_val), a.options());
  }

  // Scalar / Tensor
  if (a.dim() == 0 && b.dim() > 0) {
    at::Tensor a = full_like_cpp_scalar(b, a.item<double>());
    return trunc_div(a, b);
  }

  // Tensor / Scalar
  if (a.dim() > 0 && b.dim() == 0) {
    at::Tensor b = full_like_cpp_scalar(a, b.item<double>());
    return trunc_div(a, b);
  }

  // Tensor / Tensor
  auto res = torch::broadcast_tensors({a, b});
  at::Tensor an = res[0].contiguous();
  at::Tensor bn = res[1].contiguous();

  at::ScalarType out_dtype = at::promote_types(an.scalar_type(), bn.scalar_type());
  at::Tensor out = at::empty(an.sizes(), at::TensorOptions().dtype(out_dtype).device(an.device()));

  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(),
                                      "trunc_divide_kernel");

  int64_t tile_size = 1024;
  int64_t n = out.numel();
  int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(out.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, an, bn, out, n, tile_size);

  return out;
}

at::Tensor trunc_div_(at::Tensor& a_, const at::Tensor& b_) {
  TORCH_CHECK(a_.device().is_cuda(), "trunc_div_: only CUDA tensors supported");
  TORCH_CHECK(a_.is_contiguous(), "trunc_div_: input a_ must be contiguous for in-place op");

  //  Tensor // Scalar
  if (a_.dim() > 0 && b_.dim() == 0) {
    at::Tensor b_expand = full_like_cpp_scalar(a_, b_.item<double>());
    return trunc_div_(a_, b_expand);
  }

  //  Tensor // Tensor
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a = res[0].contiguous();
  at::Tensor b = res[1].contiguous();
  TORCH_CHECK(a_.numel() == a.numel(), "trunc_div_: in-place a_ size mismatch");

  int64_t num = a.numel();
  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(),
                                      "trunc_divide_kernel_");

  int64_t tile_size = 1024;
  int num_blocks = (num + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(a_.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, a, b, a_, num, tile_size);

  return a_;
}

constexpr const char* FLOAT_KERNEL_NAME = "float_floordiv_kernel";
constexpr const char* INT_KERNEL_NAME = "int_floordiv_kernel";
at::Tensor launch_floor_div_kernel(const at::Tensor& a_, const at::Tensor& b_, bool use_integer_kernel) {
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a = res[0].contiguous();
  at::Tensor b = res[1].contiguous();

  TORCH_CHECK(a.sizes() == b.sizes(), "Broadcasting failed.");
  TORCH_CHECK(a.device().is_cuda(), "Only CUDA supported for floor_div");

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty_like(a, at::TensorOptions().dtype(out_dtype).device(a.device()));

  std::string kernel_name = use_integer_kernel ? INT_KERNEL_NAME : FLOAT_KERNEL_NAME;
  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), kernel_name);

  int64_t tile_size = 1024;
  int64_t n = out.numel();
  int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(out.device());
  // launch Triton kernel
  f(raw_stream, num_blocks, 1, 4, 2, 0, a, b, out, n, tile_size);

  return out;
}

at::Tensor floor_div(const at::Tensor& a_, const at::Tensor& b_) {
  // ========== Scalar // Scalar ==========
  if (a_.dim() == 0 && b_.dim() == 0) {
    if (a_.is_floating_point() || b_.is_floating_point()) {
      double r = std::floor(a_.item<double>() / b_.item<double>());
      return torch::tensor(r, a_.options());
    } else {
      int64_t a_val = a_.item<int64_t>();
      int64_t b_val = b_.item<int64_t>();
      int64_t q = a_val / b_val;
      int64_t r = a_val % b_val;
      int64_t fix = (r != 0 && ((a_val < 0) != (b_val < 0))) ? 1 : 0;
      return torch::tensor(q - fix, a_.options());
    }
  }
  // ========== Tensor // Scalar ==========
  if (a_.dim() > 0 && b_.dim() == 0) {
    at::Tensor b_expand = full_like_cpp_scalar(a_, b_.item<double>());
    bool is_int = c10::isIntegralType(a_.scalar_type(), true) && c10::isIntegralType(b_.scalar_type(), true);
    return launch_floor_div_kernel(a_, b_expand, is_int);
  }
  // ========== Scalar // Tensor ==========
  if (a_.dim() == 0 && b_.dim() > 0) {
    at::Tensor a_expand = full_like_cpp_scalar(b_, a_.item<double>());
    bool is_int = c10::isIntegralType(a_.scalar_type(), true) && c10::isIntegralType(b_.scalar_type(), true);
    return launch_floor_div_kernel(a_expand, b_, is_int);
  }
  // ========== Tensor // Tensor ==========
  bool is_int = c10::isIntegralType(a_.scalar_type(), true) && c10::isIntegralType(b_.scalar_type(), true);
  return launch_floor_div_kernel(a_, b_, is_int);
}

at::Tensor launch_floor_div_kernel_(at::Tensor& a_, const at::Tensor& b_, bool use_integer_kernel) {
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a = res[0].contiguous();
  at::Tensor b = res[1].contiguous();

  TORCH_CHECK(a.sizes() == b.sizes(), "Broadcasting failed.");
  TORCH_CHECK(a.device().is_cuda(), "Only CUDA supported for floor_div");

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());

  if (a_.dtype() != out_dtype) {
    a_ = a_.to(out_dtype);
  }
  if (!a_.sizes().equals(a.sizes())) {
    a_ = at::empty(a.sizes(), a_.options().dtype(out_dtype));
    TORCH_CHECK(a_.sizes() == a.sizes(), "a_ size mismatch after broadcast.");
  }

  std::string kernel_name = use_integer_kernel ? "int_floordiv_kernel_" : "float_floordiv_kernel_";
  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), kernel_name);

  int64_t num = a.numel();
  int64_t N = a.numel() / a.size(0);
  int64_t tile_size = 1024;
  int num_blocks = (num + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(a_.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, a.to(out_dtype), b.to(out_dtype), a_, num, N, tile_size);
  return a_;
}
// ==================== main ====================
at::Tensor floor_div_(at::Tensor& a_, const at::Tensor& b_) {
  // ========== Tensor // Scalar ==========
  if (a_.dim() > 0 && b_.dim() == 0) {
    at::Tensor b_expand = full_like_cpp_scalar(a_, b_.item<double>());
    bool is_int = c10::isIntegralType(a_.scalar_type(), true) && c10::isIntegralType(b_.scalar_type(), true);
    return launch_floor_div_kernel_(a_, b_expand, is_int);
  }

  // ========== Tensor // Tensor ==========
  bool is_int = c10::isIntegralType(a_.scalar_type(), true) && c10::isIntegralType(b_.scalar_type(), true);
  return launch_floor_div_kernel_(a_, b_, is_int);
}

at::Tensor div_mode(const at::Tensor& a_,
                    const at::Tensor& b_,
                    const c10::optional<std::string>& rounding_mode) {
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a = res[0].contiguous();
  at::Tensor b = res[1].contiguous();

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());
  at::Tensor out = at::empty(a.sizes(), at::TensorOptions().dtype(out_dtype).device(a.device()));

  std::string kernel_name;
  if (rounding_mode == "floor") {
    kernel_name =
        c10::isIntegralType(a.scalar_type(), true) ? "int_floordiv_kernel" : "float_floordiv_kernel";
    ;
  } else if (rounding_mode == "trunc") {
    kernel_name = "trunc_divide_kernel";
  } else if (!rounding_mode.has_value() || rounding_mode.value() == "none") {
    kernel_name = "true_div_kernel";
  } else {
    TORCH_CHECK(false, "div_mode rounding_mode must be 'floor', 'trunc', or empty.");
  }
  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), kernel_name);

  int64_t tile_size = 1024;
  int64_t n = out.numel();
  const signed int num_blocks = (n + tile_size - 1) / tile_size;
  const int num_warps = 8;
  const int num_stages = 1;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  c10::DeviceGuard guard(out.device());
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  f(raw_stream, num_blocks, 1, 4, 2, 0, a, b, out, n, tile_size);

  return out;
}

at::Tensor div_mode_(at::Tensor& a_, const at::Tensor& b_, const c10::optional<std::string>& rounding_mode) {
  at::Tensor a = a_.contiguous();
  at::Tensor b = b_.contiguous();

  at::ScalarType out_dtype = at::promote_types(a.scalar_type(), b.scalar_type());

  if (a_.dtype() != out_dtype) {
    a_ = a_.to(out_dtype);
  }
  if (!a_.sizes().equals(a.sizes())) {
    a_ = at::empty(a.sizes(), a_.options().dtype(out_dtype));
  }

  TORCH_CHECK(a_.dtype() == out_dtype && a_.sizes() == a.sizes(),
              "a_ must match promoted dtype and broadcasted size for in-place");

  std::string kernel_name;
  if (rounding_mode == "floor") {
    kernel_name =
        c10::isIntegralType(a.scalar_type(), true) ? "int_floordiv_kernel_" : "float_floordiv_kernel_";
  } else if (rounding_mode == "trunc") {
    kernel_name = "trunc_divide_kernel_";
  } else if (!rounding_mode.has_value() || rounding_mode.value() == "none") {
    kernel_name = "true_div_kernel_";
  } else {
    TORCH_CHECK(false, "div_mode_: invalid rounding_mode");
  }

  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), kernel_name);

  TORCH_CHECK(a.dim() >= 2, "div_mode_: requires at least 2D input");
  int64_t num = a.numel();
  int64_t N = a.numel() / a.size(0);
  int64_t tile_size = 1024;
  int num_blocks = (num + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(a_.device());

  at::Tensor b_vec = b.to(out_dtype).view({num / a.size(0)});

  f(raw_stream, num_blocks, 1, 4, 2, 0, a, b_vec, a_, num, N, tile_size);

  return a_;
}
at::Tensor remainder_tt(const at::Tensor& a_, const at::Tensor& b_) {
  auto res = torch::broadcast_tensors({a_, b_});
  at::Tensor a = res[0].contiguous();
  at::Tensor b = res[1].contiguous();
  TORCH_CHECK(a.sizes() == b.sizes(), "Broadcast failed");

  at::Tensor out = at::empty_like(a, at::TensorOptions().dtype(a.scalar_type()));

  const TritonJITFunction& f =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "div.py").string(), "remainder_kernel");

  int64_t tile_size = 1024;
  int64_t n = out.numel();
  int num_blocks = (n + tile_size - 1) / tile_size;

  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  c10::DeviceGuard guard(out.device());

  f(raw_stream, num_blocks, 1, 4, 2, 0, a, b, out, n, tile_size);

  return out;
}

at::Tensor remainder_ts(const at::Tensor& a_, double b_scalar) {
  at::Tensor b = full_like_cpp_scalar(a_, b_scalar);
  return remainder_tt(a_, b);
}

at::Tensor remainder_st(double a_scalar, const at::Tensor& b_) {
  at::Tensor a = full_like_cpp_scalar(b_, a_scalar);
  return remainder_tt(a, b_);
}

at::Tensor remainder(const at::Tensor& a_, const at::Tensor& b_) {
  // Scalar % Scalar
  if (a_.dim() == 0 && b_.dim() == 0) {
    double a_val = a_.item<double>();
    double b_val = b_.item<double>();
    double r = std::fmod(a_val, b_val);
    if (r != 0.0 && ((r < 0.0) != (b_val < 0.0))) {
      r += b_val;
    }
    return torch::tensor(r, a_.options());
  }
  // Tensor % Tensor
  if (a_.dim() > 0 && b_.dim() > 0) {
    return remainder_tt(a_, b_);
  }
  // Tensor % Scalar
  if (a_.dim() > 0 && b_.dim() == 0) {
    return remainder_ts(a_, b_.item<double>());
  }
  // Scalar % Tensor
  if (a_.dim() == 0 && b_.dim() > 0) {
    return remainder_st(a_.item<double>(), b_);
  }
  TORCH_CHECK(false, "Unsupported remainder input combination.");
}

at::Tensor remainder_(at::Tensor& a_, const at::Tensor& b_) {
  TORCH_CHECK(a_.device().is_cuda(), "remainder_: only CUDA supported");

  // Tensor % Tensor
  if (a_.dim() > 0 && b_.dim() > 0) {
    at::Tensor result = remainder_tt(a_, b_);
    a_.copy_(result);
    return a_;
  }

  // Tensor % Scalar
  if (a_.dim() > 0 && b_.dim() == 0) {
    at::Tensor result = remainder_ts(a_, b_.item<double>());
    a_.copy_(result);
    return a_;
  }

  TORCH_CHECK(false, "Unsupported remainder_ input combination.");
}
}  // namespace flag_gems
