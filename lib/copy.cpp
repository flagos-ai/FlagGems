#include <c10/core/DispatchKeySet.h>
#include <vector>
#include "flag_gems/backend_utils.h"
#include "flag_gems/utils.h"
#include "torch/torch.h"
#ifdef FLAGGEMS_USE_KUNLUNXIN
#include <xpu/runtime.h>
#endif
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
  if (!backend::isOnDevice(dst) || !backend::isOnDevice(src)) return false;
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

  // Fallback checks before allocating output tensor
  if (x.is_complex() || at::isComplexType(target_dtype)) {
    return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }
  // Cross-device transfer (CPU->CUDA etc.) must fall back to PyTorch
  if (!backend::isOnDevice(x) || (device.has_value() && target_device != x.device())) {
    return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }
  if (x.scalar_type() != target_dtype) {
    return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
  }

  at::Tensor out =
      at::empty_like(x, x.options().dtype(target_dtype).device(target_device), target_memory_format);

  const int64_t numel = x.numel();
  if (numel == 0) return out;

  c10::DeviceGuard guard(target_device);

  const at::Tensor& x_linear = x;
  if (x_linear.is_contiguous() && out.is_contiguous() && numel <= std::numeric_limits<int32_t>::max()) {
#ifdef FLAGGEMS_USE_KUNLUNXIN
    // On XPU, use blocking D2D DMA instead of a Triton kernel.
    // Triton's gm2lm_v3 reads from L3/GSM (on-chip), but H2D DMA (xpu_memcpy
    // HOST_TO_DEVICE) writes only to HBM, leaving stale data in L3/GSM.
    // xpu_memcpy DEVICE_TO_DEVICE is a synchronous DMA that copies HBM->HBM
    // directly and is cache-coherent, so both src and dst are consistent.
    const uint64_t nbytes = static_cast<uint64_t>(out.nbytes());
    int ret = xpu_memcpy(out.data_ptr(), x_linear.data_ptr(), nbytes, XPU_DEVICE_TO_DEVICE);
    TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy D2D (to_copy) failed: ", ret);
    return out;
#endif
    constexpr int BLOCK_SIZE = 1024;
    const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);
    static const TritonJITFunction& kernel_linear =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, x_linear, out, numel, BLOCK_SIZE);
    return out;
  }

  // Non-contiguous path: fallback to PyTorch native implementation to avoid
  // expensive per-call GPU tensor allocations for shape/stride metadata.
  return redispatch_to_copy_fallback(x, dtype, layout, device, pin_memory, non_blocking, memory_format);
}

at::Tensor& copy_(at::Tensor& dst, const at::Tensor& src, bool non_blocking = false) {
  if (!_can_use_triton_copy(dst, src, non_blocking)) {
    return redispatch_copy_fallback(dst, src, non_blocking);
  }
  // Triton kernels do not support complex or dtype-mismatched copy
  if (dst.is_complex() || src.is_complex() || dst.scalar_type() != src.scalar_type()) {
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

  c10::DeviceGuard guard(dst.device());

  bool no_broadcast = src.sizes().equals(dst.sizes());

  if (dst.is_contiguous() && src.is_contiguous() && no_broadcast &&
      numel <= std::numeric_limits<int32_t>::max()) {
#ifdef FLAGGEMS_USE_KUNLUNXIN
    // On XPU, use blocking D2D DMA instead of a Triton kernel.
    // See comment in to_copy above for rationale.
    const uint64_t nbytes = static_cast<uint64_t>(dst.nbytes());
    int ret = xpu_memcpy(dst.data_ptr(), src.data_ptr(), nbytes, XPU_DEVICE_TO_DEVICE);
    TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy D2D (copy_) failed: ", ret);
    return dst;
#endif
    constexpr int BLOCK_SIZE = 1024;
    const unsigned int grid_x = (numel + BLOCK_SIZE - 1) / BLOCK_SIZE;
    backend::StreamType stream = backend::getCurrentStream();
    backend::RawStreamType raw_stream = backend::getRawStream(stream);
    static const TritonJITFunction& kernel_linear =
        TritonJITFunction::get_instance((utils::get_triton_src_path() / "copy.py").string(),
                                        "copy_kernel_linear");
    kernel_linear(raw_stream, grid_x, 1, 1, 4, 0, src, dst, numel, BLOCK_SIZE);
    return dst;
  }

  // Non-contiguous or broadcast path: fallback to PyTorch native implementation
  // to avoid expensive per-call GPU tensor allocations for shape/stride metadata.
  return redispatch_copy_fallback(dst, src, non_blocking);
}

}  // namespace flag_gems
