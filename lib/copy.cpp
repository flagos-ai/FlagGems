#include <c10/core/DispatchKeySet.h>
#include <vector>
#include "c10/cuda/CUDAStream.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#include "triton_jit/triton_jit_function.h"

namespace flag_gems {

using namespace triton_jit;

std::vector<int64_t> broadcasted_stride(const std::vector<int64_t>& shape,
                                        const std::vector<int64_t>& stride,
                                        const std::vector<int64_t>& target_shape) {
  int ndim_diff = target_shape.size() - shape.size();
  TORCH_CHECK(ndim_diff >= 0, "cannot broadcast to fewer dimensions");

  std::vector<int64_t> full_shape(ndim_diff, 1);
  full_shape.insert(full_shape.end(), shape.begin(), shape.end());

  std::vector<int64_t> full_stride(ndim_diff, 0);
  full_stride.insert(full_stride.end(), stride.begin(), stride.end());

  std::vector<int64_t> out_stride(target_shape.size());

  for (size_t i = 0; i < target_shape.size(); ++i) {
    if (full_shape[i] == target_shape[i]) {
      out_stride[i] = full_stride[i];
    } else if (full_shape[i] == 1) {
      out_stride[i] = 0;
    } else {
      TORCH_CHECK(false, "illegal broadcast at dim ", i);
    }
  }

  return out_stride;
}

static bool _can_use_triton_copy(const at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  if (!dst.is_cuda() || !src.is_cuda()) return false;
  if (dst.device() != src.device()) return false;
  if (non_blocking) return false;
  return true;
}

static at::Tensor& redispatch_copy_fallback(at::Tensor& dst, const at::Tensor& src, bool non_blocking) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("aten::copy_", "")
                       .typed<at::Tensor&(at::Tensor&, const at::Tensor&, bool)>();

  constexpr c10::DispatchKeySet fallback_keyset =
      c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd);

  return op.redispatch(fallback_keyset, dst, src, non_blocking);
}

static at::Tensor redispatch_to_copy_fallback(const at::Tensor& src,
                                              c10::optional<at::ScalarType> dtype,
                                              c10::optional<at::Layout> layout,
                                              c10::optional<at::Device> device,
                                              c10::optional<bool> pin_memory,
                                              bool non_blocking,
                                              c10::optional<at::MemoryFormat> memory_format) {
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("aten::_to_copy", "")
                       .typed<at::Tensor(const at::Tensor&,
                                         c10::optional<at::ScalarType>,
                                         c10::optional<at::Layout>,
                                         c10::optional<at::Device>,
                                         c10::optional<bool>,
                                         bool,
                                         c10::optional<at::MemoryFormat>)>();

  constexpr c10::DispatchKeySet fallback_keyset =
      c10::DispatchKeySet(c10::DispatchKey::CompositeExplicitAutograd);

  return op.redispatch(fallback_keyset, src, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

at::Tensor to_copy(const at::Tensor& x,
                   c10::optional<at::ScalarType> dtype = c10::nullopt,
                   c10::optional<at::Layout> layout = c10::nullopt,
                   c10::optional<at::Device> device = c10::nullopt,
                   c10::optional<bool> pin_memory = c10::nullopt,
                   bool non_blocking = false,
                   c10::optional<at::MemoryFormat> memory_format = c10::nullopt) {
  TORCH_WARN("[flag_gems][to_copy] gems::to_copy");
  TORCH_CHECK(x.layout() == at::Layout::Strided, "Only strided tensors are supported");
  TORCH_CHECK(!x.is_quantized(), "Quantized tensors are not supported");
  if (layout.has_value()) {
    TORCH_CHECK(layout.value() == x.layout(), "to_copy: layout conversion is not supported");
  }
  TORCH_CHECK(!pin_memory.has_value(), "to_copy: pin_memory is not supported");
  TORCH_CHECK(!non_blocking, "to_copy: non_blocking copy is not supported");

  auto target_dtype = dtype.has_value() ? dtype.value() : x.scalar_type();
  auto target_device = device.has_value() ? device.value() : x.device();
  auto target_memory_format = memory_format.has_value() ? memory_format.value() : at::MemoryFormat::Preserve;

  at::Tensor out =
      at::empty_like(x, x.options().dtype(target_dtype).device(target_device), target_memory_format);

  // if (!_can_use_triton_copy(out, x, non_blocking)) {
  //   return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
  // }

  const int64_t numel = x.numel();
  if (numel == 0) return out;

  constexpr int BLOCK_SIZE = 1024;
  const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  c10::DeviceGuard guard(target_device);
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  // at::Tensor x_linear = (x.scalar_type() != target_dtype) ? x.to(target_dtype) : x;
  at::Tensor x_linear = x;
  if (x.scalar_type() != target_dtype) {
    return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }
  if (x_linear.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
    const TritonJITFunction& kernel_linear =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, x_linear, out, numel, BLOCK_SIZE);
    return out;
  }

  std::vector<int64_t> task_shape(out.sizes().begin(), out.sizes().end());
  int NDIMS = task_shape.size();

  std::vector<int64_t> src_stride =
      broadcasted_stride(std::vector<int64_t>(x_linear.sizes().begin(), x_linear.sizes().end()),
                         std::vector<int64_t>(x_linear.strides().begin(), x_linear.strides().end()),
                         task_shape);
  std::vector<int64_t> dst_stride =
      broadcasted_stride(std::vector<int64_t>(out.sizes().begin(), out.sizes().end()),
                         std::vector<int64_t>(out.strides().begin(), out.strides().end()),
                         task_shape);

  const TritonJITFunction& kernel_nd =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(), "copy_kernel_nd");
  kernel_nd(raw_stream,
            grid_x,
            1,
            1,
            4,
            0,
            x_linear,
            out,
            torch::tensor(task_shape, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            torch::tensor(src_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            torch::tensor(dst_stride, torch::TensorOptions().dtype(torch::kInt64).device(out.device())),
            numel,
            NDIMS,
            BLOCK_SIZE);

  return out;
}

at::Tensor& copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking = false) {
  TORCH_WARN("[flag_gems][copy_] gems::copy_");
  if (!_can_use_triton_copy(dst, src, non_blocking)) {
    return redispatch_copy_fallback(dst, src, non_blocking);
  }
  TORCH_CHECK(!dst._is_zerotensor(), "ZeroTensors are immutable");
  if (src._is_zerotensor()) {
    dst.zero_();
    return dst;
  }

  if (dst.data_ptr() == src.data_ptr()) return dst;

  TORCH_CHECK(src.sizes().size() <= dst.sizes().size(), "src cannot be broadcasted to dst");
  for (size_t i = 0; i < src.dim(); ++i) {
    TORCH_CHECK(src.size(i) == dst.size(dst.dim() - src.dim() + i) || src.size(i) == 1,
                "src cannot be broadcasted to dst");
  }

  const int64_t numel = dst.numel();

  constexpr int BLOCK_SIZE = 1024;
  const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;

  c10::DeviceGuard guard(dst.device());
  c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream();
  CUstream raw_stream = static_cast<CUstream>(stream.stream());

  bool no_broadcast = src.sizes().equals(dst.sizes());

  if (dst.is_contiguous() && src.is_contiguous() && no_broadcast &&
      numel <= std::numeric_limits<int32_t>::max()) {
    const TritonJITFunction& kernel_linear =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, src, dst, numel, BLOCK_SIZE);
    return dst;
  }

  std::vector<int64_t> task_shape(dst.sizes().begin(), dst.sizes().end());
  int NDIMS = task_shape.size();

  std::vector<int64_t> src_stride =
      broadcasted_stride(std::vector<int64_t>(src.sizes().begin(), src.sizes().end()),
                         std::vector<int64_t>(src.strides().begin(), src.strides().end()),
                         task_shape);
  std::vector<int64_t> dst_stride =
      broadcasted_stride(std::vector<int64_t>(dst.sizes().begin(), dst.sizes().end()),
                         std::vector<int64_t>(dst.strides().begin(), dst.strides().end()),
                         task_shape);

  const TritonJITFunction& kernel_nd =
      TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(), "copy_kernel_nd");
  kernel_nd(raw_stream,
            grid_x,
            1,
            1,
            4,
            0,
            src,
            dst,
            torch::tensor(task_shape, torch::TensorOptions().dtype(torch::kInt64).device(dst.device())),
            torch::tensor(src_stride, torch::TensorOptions().dtype(torch::kInt64).device(dst.device())),
            torch::tensor(dst_stride, torch::TensorOptions().dtype(torch::kInt64).device(dst.device())),
            numel,
            NDIMS,
            BLOCK_SIZE);
  return dst;
}

}  // namespace flag_gems
