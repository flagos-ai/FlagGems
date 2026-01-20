#include "flag_gems/accuracy_utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>
#include <sstream>

namespace flag_gems::accuracy_utils {

bool TO_CPU = false;

float resolution_for_dtype(c10::ScalarType dtype) {
  switch (dtype) {
    case c10::ScalarType::Byte:
    case c10::ScalarType::Char:
    case c10::ScalarType::Short:
    case c10::ScalarType::Int:
    case c10::ScalarType::Long:
      return 0.0;

    case c10::ScalarType::Half:
      return 1e-3;
    case c10::ScalarType::Float:
      return 1.3e-6;
    case c10::ScalarType::BFloat16:
      return 0.016;
    case c10::ScalarType::Double:
      return 1e-7;

    case c10::ScalarType::ComplexFloat:
      return 1.3e-6;
    case c10::ScalarType::ComplexDouble:
      return 1e-7;
#if defined(C10_HAS_FLOAT8)
    case c10::ScalarType::Float8_e4m3fn:
    case c10::ScalarType::Float8_e4m3fnuz:
    case c10::ScalarType::Float8_e5m2:
    case c10::ScalarType::Float8_e5m2fnuz:
      return 1e-3;
#endif

    default:
      TORCH_CHECK(false, "Unsupported dtype in resolution_for_dtype: ", c10::toString(dtype));
  }
}

torch::Tensor to_reference(torch::Tensor inp, bool upcast) {
  if (!inp.defined()) {
    return torch::Tensor();
  }

  torch::Tensor ref_inp = inp;

  if (TO_CPU) {
    ref_inp = ref_inp.to(torch::kCPU);
  }

  if (upcast) {
    if (ref_inp.is_complex()) {
      ref_inp = ref_inp.to(torch::kComplexDouble);
    } else {
      ref_inp = ref_inp.to(torch::kDouble);
    }
  }

  return ref_inp;
}

torch::Tensor to_cpu(torch::Tensor res, const torch::Tensor& ref) {
  if (TO_CPU) {
    TORCH_CHECK(ref.device().is_cpu(),
                "to_cpu: reference tensor must be on CPU when TO_CPU is enabled, "
                "but got device = ",
                ref.device().str());
    res = res.to(torch::kCPU);
  }
  return res;
}

static std::pair<torch::Tensor, torch::Tensor> _maybe_move_to_cpu(torch::Tensor res, torch::Tensor ref) {
  if (!(res.is_cuda() && ref.is_cuda())) {
    return {res, ref};
  }

  const int64_t required = res.numel() * static_cast<int64_t>(res.element_size());

  int64_t free_mem = -1;

  try {
    size_t free_mem_u = 0, total_mem_u = 0;
    c10::cuda::CUDAGuard device_guard(res.device());
    cudaMemGetInfo(&free_mem_u, &total_mem_u);
    free_mem = static_cast<int64_t>(free_mem_u);
  } catch (...) {
    free_mem = -1;
  }

  constexpr int64_t HUGE_TENSOR_BYTES = int64_t(1) << 30;  // 1 GiB

  if ((free_mem >= 0 && required >= free_mem) || (required >= HUGE_TENSOR_BYTES)) {
    return {res.cpu(), ref.cpu()};
  }

  return {res, ref};
}

CheckCloseResult gems_assert_close(torch::Tensor res,
                                   torch::Tensor ref,
                                   c10::ScalarType dtype,
                                   bool equal_nan,
                                   int64_t reduce_dim,
                                   float atol) {
  res = to_cpu(res, ref);

  if (dtype == c10::ScalarType::Undefined) {
    // dtype = c10::kFloat;
    dtype = res.scalar_type();
  }

  TORCH_CHECK(res.scalar_type() == dtype,
              "gems_assert_close: res dtype mismatch, expect ",
              c10::toString(dtype),
              ", got ",
              c10::toString(res.scalar_type()));

  ref = ref.to(dtype);

  std::tie(res, ref) = _maybe_move_to_cpu(res, ref);

  const float rtol = resolution_for_dtype(dtype);

  const float scaled_atol = atol * reduce_dim;

  bool ok = torch::allclose(res, ref, rtol, scaled_atol, equal_nan);

  if (ok) {
    return {true, ""};
  }

  auto diff = (res - ref).abs();
  float real_atol = diff.max().item<float>();
  auto denom = ref.abs() + 1e-12;
  float real_rtol = (diff / denom).max().item<float>();

  std::ostringstream oss;
  oss << "gems_assert_close failed\n"
      << "dtype      : " << c10::toString(dtype) << "\n"
      << "used atol  : " << scaled_atol << "\n"
      << "used rtol  : " << rtol << "\n"
      << "real atol  : " << real_atol << "\n"
      << "real rtol  : " << real_rtol << "\n";

  return {false, oss.str()};
}

CheckCloseResult gems_assert_equal(torch::Tensor res, torch::Tensor ref, bool equal_nan) {
  res = to_cpu(res, ref);

  bool ok = torch::allclose(res, ref, 0.0, 0.0, equal_nan);

  if (ok) {
    return {true, ""};
  }

  auto diff = (res - ref).abs().max().item<float>();
  std::ostringstream oss;
  oss << "gems_assert_equal failed\n"
      << "equal_nan : " << equal_nan << "\n"
      << "max diff : " << diff << "\n";

  return {false, oss.str()};
}

}  // namespace flag_gems::accuracy_utils
