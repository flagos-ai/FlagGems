#pragma once

#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <torch/torch.h>

// Backend-specific includes and type definitions
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
namespace flag_gems {
namespace backend {
  using StreamType = c10::cuda::CUDAStream;
  using RawStreamType = CUstream;
}  // namespace backend
}  // namespace flag_gems
#elif defined(FLAGGEMS_USE_NPU)
#include <acl/acl.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
namespace flag_gems {
namespace backend {
  using StreamType = c10_npu::NPUStream;
  using RawStreamType = aclrtStream;
}  // namespace backend
}  // namespace flag_gems
#elif defined(FLAGGEMS_USE_MUSA)
#include <musa_runtime.h>
#include "torch_musa/csrc/core/MUSAStream.h"
namespace flag_gems {
namespace backend {
  using StreamType = c10::musa::MUSAStream;
  using RawStreamType = musaStream_t;
}  // namespace backend
}  // namespace flag_gems
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
#include <xpu/runtime.h>
namespace flag_gems {
namespace backend {
  // Kunlunxin XRE3 native path: the "stream" is a raw XPUStream (void*) which
  // doubles as the raw launch handle for triton_jit's Kunlunxin backend.
  using StreamType = XPUStream;
  using RawStreamType = XPUStream;
}  // namespace backend
}  // namespace flag_gems
#endif

namespace flag_gems {
namespace backend {

  // Get the current stream for the given device
  inline StreamType getCurrentStream(const at::Device& device) {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return c10::cuda::getCurrentCUDAStream(device.index());
#elif defined(FLAGGEMS_USE_NPU)
    return c10_npu::getCurrentNPUStream(device.index());
#elif defined(FLAGGEMS_USE_MUSA)
    return c10::musa::getCurrentMUSAStream(device.index());
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    (void)device;
    // TODO: XRE3 has no thread-local "current stream" API; nullptr = default stream.
    return nullptr;
#else
#error \
    "No backend defined. Define one of: FLAGGEMS_USE_CUDA, FLAGGEMS_USE_IX, FLAGGEMS_USE_NPU, FLAGGEMS_USE_MUSA, FLAGGEMS_USE_KUNLUNXIN"
#endif
  }

  // Get the current stream for the default device
  inline StreamType getCurrentStream() {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return c10::cuda::getCurrentCUDAStream();
#elif defined(FLAGGEMS_USE_NPU)
    return c10_npu::getCurrentNPUStream();
#elif defined(FLAGGEMS_USE_MUSA)
    return c10::musa::getCurrentMUSAStream();
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    // TODO: XRE3 has no thread-local "current stream" API; nullptr = default stream.
    return nullptr;
#else
#error "No backend defined"
#endif
  }

  // Get the raw stream from a typed stream (for passing to triton_jit)
  inline RawStreamType getRawStream(const StreamType& stream) {
#if defined(FLAGGEMS_USE_KUNLUNXIN)
    return stream;  // already raw XPUStream (void*)
#else
    return stream.stream();
#endif
  }

  // Check if tensor is on the correct device type for this backend
  inline void checkDeviceType(const at::Tensor& tensor, const char* tensor_name) {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    TORCH_CHECK(tensor.is_cuda(), tensor_name, " must be on CUDA device, but got ", tensor.device());
#elif defined(FLAGGEMS_USE_NPU)
    TORCH_CHECK(tensor.is_privateuseone(), tensor_name, " must be on NPU device, but got ", tensor.device());
#elif defined(FLAGGEMS_USE_MUSA)
    TORCH_CHECK(tensor.is_privateuseone(), tensor_name, " must be on MUSA device, but got ", tensor.device());
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    TORCH_CHECK(tensor.is_privateuseone(),
                tensor_name,
                " must be on Kunlunxin (XPU) device, but got ",
                tensor.device());
#else
#error "No backend defined"
#endif
  }

  // Check if tensor is on the correct device type (returns bool instead of throwing)
  inline bool isOnDevice(const at::Tensor& tensor) {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return tensor.is_cuda();
#elif defined(FLAGGEMS_USE_NPU)
    return tensor.is_privateuseone();
#elif defined(FLAGGEMS_USE_MUSA)
    return tensor.is_privateuseone();
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    return tensor.is_privateuseone();
#else
#error "No backend defined"
#endif
  }

  // Get the device type string for error messages
  inline const char* getDeviceTypeName() {
#if defined(FLAGGEMS_USE_CUDA)
    return "CUDA";
#elif defined(FLAGGEMS_USE_IX)
    return "IX (CUDA-compatible)";
#elif defined(FLAGGEMS_USE_NPU)
    return "NPU";
#elif defined(FLAGGEMS_USE_MUSA)
    return "MUSA";
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    return "Kunlunxin";
#else
#error "No backend defined"
#endif
  }

  // Get the torch device type used by the active backend.
  inline at::DeviceType getBackendDeviceType() {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return at::kCUDA;
#elif defined(FLAGGEMS_USE_NPU) || defined(FLAGGEMS_USE_MUSA) || defined(FLAGGEMS_USE_KUNLUNXIN)
    return at::kPrivateUse1;
#else
#error "No backend defined"
#endif
  }

  // Get the current device index for the active backend.
  inline c10::DeviceIndex getCurrentDeviceIndex() {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return at::cuda::current_device();
#elif defined(FLAGGEMS_USE_MUSA)
    return c10::musa::current_device();
#elif defined(FLAGGEMS_USE_NPU)
    return 0;  // TODO: NPU current device query
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    int dev = 0;
    xpu_current_device(&dev);
    return static_cast<c10::DeviceIndex>(dev);
#else
    return 0;
#endif
  }

  // Get the current torch device for the active backend.
  inline at::Device getCurrentDevice() {
    return at::Device(getBackendDeviceType(), getCurrentDeviceIndex());
  }

  // Get the default torch device for tensors allocated by this backend.
  inline at::Device getDefaultDevice(int index = 0) {
    return at::Device(getBackendDeviceType(), static_cast<c10::DeviceIndex>(index));
  }

  // Check if the backend device is available.
  inline bool isDeviceAvailable() {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    return torch::cuda::is_available();
#elif defined(FLAGGEMS_USE_NPU)
    return torch::custom_class_available("npu");
#elif defined(FLAGGEMS_USE_MUSA)
    return true;
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    return true;
#else
    return false;
#endif
  }

  // Synchronize the backend device.
  inline void synchronize() {
#if defined(FLAGGEMS_USE_CUDA) || defined(FLAGGEMS_USE_IX)
    torch::cuda::synchronize();
#elif defined(FLAGGEMS_USE_NPU)
    // NPU sync if needed
#elif defined(FLAGGEMS_USE_MUSA)
    // MUSA sync if needed
#elif defined(FLAGGEMS_USE_KUNLUNXIN)
    xpu_wait(NULL);  // XRE3 native: wait all pending ops on default stream
#endif
  }

}  // namespace backend
}  // namespace flag_gems
