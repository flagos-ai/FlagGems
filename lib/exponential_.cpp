#include <ATen/ATen.h>
#include <ATen/Generator.h>
#include <ATen/core/Generator.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <torch/torch.h>
#include <iostream>
#include "c10/cuda/CUDAStream.h"
#include "flag_gems/operators.h"
#include "flag_gems/utils.h"
#include "triton_jit/triton_jit_function.h"
namespace flag_gems {
using namespace triton_jit;
enum class FloatType { Float32, Float64, Float16, BFloat16 };
template <typename T>
double get_epsilon() {
  return std::numeric_limits<T>::epsilon();
}
std::string get_type_name(FloatType type) {
  static const std::map<FloatType, std::string> type_names = {
      { FloatType::Float32,  "float32"},
      { FloatType::Float64,  "float64"},
      { FloatType::Float16,  "float16"},
      {FloatType::BFloat16, "bfloat16"}
  };
  return type_names.at(type);
}

double get_epsilon(FloatType type) {
  switch (type) {
    case FloatType::Float32:
      return get_epsilon<float>();
    case FloatType::Float64:
      return get_epsilon<double>();
    case FloatType::Float16:
      return 0.0009765625;
    case FloatType::BFloat16:
      return 0.0078125;
    default:
      throw std::invalid_argument("Unsupported floating point type");
  }
}
FloatType dtype_to_floattype(torch::Dtype dtype) {
  if (dtype == torch::kFloat32) return FloatType::Float32;
  if (dtype == torch::kFloat64) return FloatType::Float64;
  if (dtype == torch::kFloat16) return FloatType::Float16;
  throw std::invalid_argument("Unsupported dtype");
}
std::string get_vendor_name_simulated() {
  return "nvidia";
}
at::Device get_current_torch_device() {
  if (torch::cuda::is_available()) {
    return at::Device(at::kCUDA, at::cuda::current_device());
  } else {
    return at::Device(at::kCPU);
  }
}
std::pair<uint64_t, uint64_t> philox_backend_seed_offset(
    int64_t increment, c10::optional<at::Generator> generator_opt = c10::nullopt) {
  at::Generator gen;
  if (generator_opt.has_value()) {
    gen = generator_opt.value();
  } else {
    at::Device device = get_current_torch_device();
    gen = at::globalContext().defaultGenerator(device);
  }

  at::Tensor state_copy = gen.get_state();
  at::Tensor state_int64_view = state_copy.view(at::kLong);

  int64_t c0, c1;
  std::string vendor_name = get_vendor_name_simulated();

  if (vendor_name == "kunlunxin" || vendor_name == "aipu") {
    TORCH_CHECK(state_int64_view.numel() >= 2,
                "Generator state is too small for 'kunlunxin' or 'aipu' vendor.");
    c0 = state_int64_view[state_int64_view.numel() - 2].item<int64_t>();
    c1 = state_int64_view[state_int64_view.numel() - 1].item<int64_t>();
  } else {
    TORCH_CHECK(state_int64_view.numel() >= 2, "Generator state is too small for default vendor.");
    c0 = state_int64_view[0].item<int64_t>();
    c1 = state_int64_view[1].item<int64_t>();
  }
  const int64_t seed = c0;
  const int64_t offset = c1;
  increment = (increment + 3) / 4 * 4;

  gen.set_state(state_copy);
  return std::make_pair(seed, offset);
}
at::Tensor &exponential_(at::Tensor &self, double lambd, c10::optional<at::Generator> gen) {
  torch::Dtype dtype = self.scalar_type();
  torch::Device device = self.device();

  bool inplace = self.is_contiguous();

  AT_ASSERT(dtype == torch::kFloat16 || dtype == torch::kBFloat16 || dtype == torch::kFloat32 ||
                dtype == torch::kFloat64,
            "Unsupported dtype");
  TORCH_CHECK(lambd > 0.0, "exponential_ requires lambd > 0.0, but got ", lambd);
  TORCH_CHECK(self.is_cuda(), "exponential_ currently only supports CUDA tensors");
  bool is_double = (dtype == torch::kFloat64);

  const int UNROLL = is_double ? 2 : 4;

  const int64_t N = self.numel();
  const int64_t increment = (N + UNROLL - 1) / UNROLL;

  auto [philox_seed, philox_offset] = philox_backend_seed_offset(increment, gen);

  at::Tensor x_;
  if (inplace) {
    x_ = self;
  } else {
    x_ = at::empty(self.sizes(), at::TensorOptions().dtype(dtype).device(device));
  }
  constexpr int64_t BLOCK = 128;
  unsigned grid_x = (N + BLOCK * UNROLL - 1) / (BLOCK * UNROLL);

  const TritonJITFunction &f = TritonJITFunction::get_instance(
      std::string(utils::get_flag_gems_src_path() / "ops" / "exponential_.py"),
      "fused_exponential_kernel");

  c10::DeviceGuard guard(x_.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());
  f(raw_stream,
    grid_x,
    /* grid_y = */ 1,
    /* grid_z = */ 1,
    /* num_warps = */ 8,
    /* num_stages = */ 1,
    x_,
    N,
    is_double,
    lambd,
    get_epsilon(dtype_to_floattype(dtype)),
    static_cast<int64_t>(philox_seed),
    static_cast<int64_t>(philox_offset),
    128);
  if (!inplace) {
    self.copy_(x_);
    return self;
  }
  return self;
}
}  // namespace flag_gems
