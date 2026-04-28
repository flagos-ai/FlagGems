// XPU (PrivateUse1) tensor allocation and device registration using native XRE3.
//
// torch_xmlir previously provided these registrations for the PrivateUse1
// backend (via `import torch_xmlir`). Now that FlagGems no longer builds
// against torch_xmlir headers, this file provides the minimal C++ registration
// needed so that C++ code (and ctests) can allocate/use XPU tensors.
//
// Only compiled for the KUNLUNXIN backend.

#ifdef FLAGGEMS_USE_KUNLUNXIN

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Allocator.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <torch/library.h>
#include <xpu/runtime.h>

namespace flag_gems {
namespace xpu {

  // ---------------------------------------------------------------------------
  // DeviceGuardImpl — required by c10::DeviceGuard used in copy_.cpp / to_copy
  // ---------------------------------------------------------------------------

  struct XpuDeviceGuardImpl final : public c10::impl::DeviceGuardImplInterface {
    c10::DeviceType type() const override {
      return c10::DeviceType::PrivateUse1;
    }

    c10::Device exchangeDevice(c10::Device d) const override {
      int prev = 0;
      xpu_current_device(&prev);
      xpu_set_device(static_cast<int>(d.index()));
      return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(prev));
    }

    c10::Device getDevice() const override {
      int dev = 0;
      xpu_current_device(&dev);
      return c10::Device(c10::DeviceType::PrivateUse1, static_cast<c10::DeviceIndex>(dev));
    }

    void setDevice(c10::Device d) const override {
      xpu_set_device(static_cast<int>(d.index()));
    }

    void uncheckedSetDevice(c10::Device d) const noexcept override {
      xpu_set_device(static_cast<int>(d.index()));
    }

    // XRE3 has no per-thread stream concept; return a default stream stub.
    c10::Stream getStream(c10::Device d) const noexcept override {
      return c10::Stream(c10::Stream::UNSAFE, d, 0);
    }

    c10::Stream exchangeStream(c10::Stream s) const noexcept override {
      return s;  // no-op: XRE3 uses implicit default stream
    }

    c10::DeviceIndex deviceCount() const noexcept override {
      int n = 0;
      xpu_device_count(&n);
      return static_cast<c10::DeviceIndex>(n);
    }
  };

  static XpuDeviceGuardImpl g_xpu_guard_impl;
  static c10::impl::DeviceGuardImplRegistrar g_xpu_guard_registrar(c10::DeviceType::PrivateUse1,
                                                                   &g_xpu_guard_impl);

  // ---------------------------------------------------------------------------
  // Allocator
  // ---------------------------------------------------------------------------

  struct XpuAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t nbytes) override {
      if (nbytes == 0) {
        return c10::DataPtr(nullptr,
                            nullptr,
                            &c10::detail::deleteNothing,
                            c10::Device(c10::DeviceType::PrivateUse1, 0));
      }
      void* data = nullptr;
      int ret = xpu_malloc(&data, static_cast<uint64_t>(nbytes));
      TORCH_CHECK(ret == XPU_SUCCESS, "xpu_malloc(", nbytes, " bytes) failed with code ", ret);
      int dev = 0;
      xpu_current_device(&dev);
      return c10::DataPtr(
          data,
          data,
          [](void* p) {
            if (p) xpu_free(p);
          },
          c10::Device(c10::DeviceType::PrivateUse1, dev));
    }

    void copy_data(void* dest, const void* src, std::size_t count) const override {
      int ret = xpu_memcpy(dest, src, static_cast<uint64_t>(count), XPU_DEVICE_TO_DEVICE);
      TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy D2D failed with code ", ret);
    }

    c10::DeleterFnPtr raw_deleter() const override {
      return [](void* p) {
        if (p) xpu_free(p);
      };
    }
  };

  static XpuAllocator g_xpu_allocator;

  // ---------------------------------------------------------------------------
  // Helper: build a contiguous XPU tensor from allocated memory
  // ---------------------------------------------------------------------------

  static at::Tensor make_xpu_tensor(at::IntArrayRef size, at::ScalarType scalar_type, int /*device_index*/) {
    int64_t numel = 1;
    for (auto s : size) numel *= s;
    int64_t nbytes = numel * at::elementSize(scalar_type);

    c10::DataPtr data_ptr = g_xpu_allocator.allocate(nbytes > 0 ? nbytes : 0);

    auto storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t {},
                                                         nbytes,
                                                         std::move(data_ptr),
                                                         /*allocator=*/&g_xpu_allocator,
                                                         /*resizable=*/true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage),
                                                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                                           c10::scalarTypeToTypeMeta(scalar_type));

    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
    return tensor;
  }

  // ---------------------------------------------------------------------------
  // aten::empty.memory_format
  // ---------------------------------------------------------------------------

  at::Tensor xpu_empty(at::IntArrayRef size,
                       c10::optional<at::ScalarType> dtype,
                       c10::optional<at::Layout> layout,
                       c10::optional<at::Device> device,
                       c10::optional<bool> pin_memory,
                       c10::optional<at::MemoryFormat> memory_format) {
    TORCH_CHECK(!pin_memory.value_or(false), "XPU does not support pinned memory");
    TORCH_CHECK(!layout.has_value() || *layout == at::Layout::Strided, "XPU only supports strided layout");

    auto scalar_type = dtype.value_or(at::kFloat);
    int dev = device.has_value() ? device->index() : 0;

    at::Tensor t = make_xpu_tensor(size, scalar_type, dev);

    auto mf = memory_format.value_or(at::MemoryFormat::Contiguous);
    if (mf != at::MemoryFormat::Contiguous) {
      t.unsafeGetTensorImpl()->empty_tensor_restride(mf);
    }
    return t;
  }

  // ---------------------------------------------------------------------------
  // aten::empty_strided
  // ---------------------------------------------------------------------------

  at::Tensor xpu_empty_strided(at::IntArrayRef size,
                               at::IntArrayRef stride,
                               c10::optional<at::ScalarType> dtype,
                               c10::optional<at::Layout> layout,
                               c10::optional<at::Device> device,
                               c10::optional<bool> pin_memory) {
    TORCH_CHECK(!pin_memory.value_or(false), "XPU does not support pinned memory");

    auto scalar_type = dtype.value_or(at::kFloat);
    int dev = device.has_value() ? device->index() : 0;

    int64_t max_offset = 0;
    for (size_t i = 0; i < size.size(); ++i) {
      if (size[i] > 0) {
        max_offset += (size[i] - 1) * std::abs(stride[i]);
      }
    }
    int64_t nbytes = (max_offset + 1) * at::elementSize(scalar_type);

    c10::DataPtr data_ptr = g_xpu_allocator.allocate(nbytes > 0 ? nbytes : 0);

    auto storage = c10::make_intrusive<c10::StorageImpl>(c10::StorageImpl::use_byte_size_t {},
                                                         nbytes,
                                                         std::move(data_ptr),
                                                         /*allocator=*/&g_xpu_allocator,
                                                         /*resizable=*/true);

    auto tensor = at::detail::make_tensor<c10::TensorImpl>(std::move(storage),
                                                           c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
                                                           c10::scalarTypeToTypeMeta(scalar_type));

    tensor.unsafeGetTensorImpl()->set_sizes_and_strides(size, stride);
    return tensor;
  }

  // ---------------------------------------------------------------------------
  // aten::normal_ — fill tensor with normal distribution (CPU fallback + H2D)
  // ---------------------------------------------------------------------------

  at::Tensor& xpu_normal_(at::Tensor& self, double mean, double std, c10::optional<at::Generator> gen) {
    const uint64_t nbytes = static_cast<uint64_t>(self.nbytes());

    // Generate random values on CPU.
    auto cpu_tensor = at::empty(self.sizes(), self.options().device(at::kCPU).dtype(self.dtype()));
    at::native::normal_(cpu_tensor, mean, std, gen);

    // Use blocking xpu_memcpy for H2D.  The synchronous path is cache-coherent on
    // KunlunXin: the firmware's software-copy route writes through the XPU L2
    // cache, so subsequent device kernels reading the same address see fresh data.
    // xpu_memcpy_async (even with pinned memory) uses a DMA engine that bypasses
    // the XPU L2 cache, leaving stale cache lines and producing wrong kernel reads.
    int ret = xpu_memcpy(self.data_ptr(), cpu_tensor.data_ptr(), nbytes, XPU_HOST_TO_DEVICE);
    TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy H2D (normal_) failed: ", ret);
    return self;
  }

  // ---------------------------------------------------------------------------
  // aten::as_strided — create a non-contiguous view (metadata-only, no data copy)
  // ---------------------------------------------------------------------------

  at::Tensor xpu_as_strided(const at::Tensor& self,
                            at::IntArrayRef size,
                            at::IntArrayRef stride,
                            c10::optional<int64_t> storage_offset) {
    auto result = self.detach();
    auto* impl = result.unsafeGetTensorImpl();
    impl->set_storage_offset(storage_offset.value_or(self.storage_offset()));
    impl->set_sizes_and_strides(size, stride);
    return result;
  }

  // ---------------------------------------------------------------------------
  // aten::_copy_from — low-level H2D / D2H / D2D copy
  //
  // Fast path: contiguous, same dtype, same byte count → raw xpu_memcpy.
  // General path: bring src to CPU, use CPU copy_ (handles dtype conversion and
  // broadcasting), then H2D the result back to dst if needed.
  // ---------------------------------------------------------------------------

  at::Tensor xpu_copy_from(const at::Tensor& src, const at::Tensor& dst, bool non_blocking) {
    const bool src_xpu = src.device().type() == c10::DeviceType::PrivateUse1;
    const bool dst_xpu = dst.device().type() == c10::DeviceType::PrivateUse1;

    // Fast path: contiguous, same dtype, same storage byte count → raw memcpy.
    if (src.scalar_type() == dst.scalar_type() && src.is_contiguous() && dst.is_contiguous() &&
        src.nbytes() == dst.nbytes()) {
      XPUMemcpyKind kind;
      if (!src_xpu && dst_xpu)
        kind = XPU_HOST_TO_DEVICE;
      else if (src_xpu && !dst_xpu)
        kind = XPU_DEVICE_TO_HOST;
      else
        kind = XPU_DEVICE_TO_DEVICE;
      int ret = xpu_memcpy(dst.data_ptr(), src.data_ptr(), static_cast<uint64_t>(src.nbytes()), kind);
      TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy (_copy_from) failed: ", ret);
      return dst;
    }

    // General path: round-trip through CPU for dtype conversion / broadcasting /
    // non-contiguous layouts.
    //
    // 1. D2H: copy src storage to a CPU buffer.
    at::Tensor cpu_src;
    std::vector<char> d2h_buf;  // backing store for cpu_src when src is on XPU
    if (src_xpu) {
      size_t storage_bytes = src.storage().nbytes();
      d2h_buf.resize(storage_bytes);
      int ret = xpu_memcpy(d2h_buf.data(), src.storage().data(), storage_bytes, XPU_DEVICE_TO_HOST);
      TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy D2H (_copy_from src) failed: ", ret);
      // Build a CPU tensor view with the same size/stride/offset as src.
      char* base = d2h_buf.data() + src.storage_offset() * src.element_size();
      cpu_src = at::from_blob(static_cast<void*>(base),
                              src.sizes(),
                              src.strides(),
                              at::TensorOptions().dtype(src.scalar_type()).device(at::kCPU));
    } else {
      cpu_src = src;
    }

    // 2. CPU-to-CPU copy (handles dtype conversion + broadcasting).
    //
    // Mirror dst's exact strides on CPU so that after copy_(), the physical
    // byte layout of cpu_dst matches dst's storage exactly. This lets us
    // H2D the raw storage bytes without needing dst to be contiguous.
    at::Tensor cpu_dst;
    if (dst_xpu) {
      cpu_dst =
          at::empty_strided(dst.sizes(), dst.strides(), dst.options().device(at::kCPU).dtype(dst.dtype()));
    } else {
      cpu_dst = dst;
    }
    cpu_dst.copy_(cpu_src);

    // 3. H2D: copy CPU result back into dst's storage on XPU.
    //    cpu_dst has the same sizes/strides/storage_offset==0 as dst, so their
    //    storage() regions have an identical layout.  A single flat memcpy of
    //    the storage bytes is sufficient.
    if (dst_xpu) {
      size_t storage_bytes = dst.storage().nbytes();
      TORCH_CHECK(cpu_dst.storage().nbytes() == storage_bytes,
                  "_copy_from: cpu_dst/dst storage size mismatch (",
                  cpu_dst.storage().nbytes(),
                  " vs ",
                  storage_bytes,
                  ")");
      int ret = xpu_memcpy(const_cast<void*>(dst.storage().data()),
                           cpu_dst.storage().data(),
                           storage_bytes,
                           XPU_HOST_TO_DEVICE);
      TORCH_CHECK(ret == XPU_SUCCESS, "xpu_memcpy H2D (_copy_from dst) failed: ", ret);
    }
    return dst;
  }

  // ---------------------------------------------------------------------------
  // Registration — happens at library load via static init
  // ---------------------------------------------------------------------------

  TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("empty.memory_format", TORCH_FN(xpu_empty));
    m.impl("empty_strided", TORCH_FN(xpu_empty_strided));
    m.impl("normal_", TORCH_FN(xpu_normal_));
    m.impl("as_strided", TORCH_FN(xpu_as_strided));
    m.impl("_copy_from", TORCH_FN(xpu_copy_from));
  }

  // ---------------------------------------------------------------------------
  // Dynamic re-registration — call this after any late-loaded library (e.g.
  // torch_xmlir) may have overridden our PrivateUse1 kernels.  Stores the
  // Library object as a process-lifetime global so the registrations persist.
  // ---------------------------------------------------------------------------

  // Function-pointer used by the dynamic re-registrar (cannot be a lambda).
  static void do_register_xpu_kernels(torch::Library& m) {
    m.impl("empty.memory_format", TORCH_FN(xpu_empty));
    m.impl("empty_strided", TORCH_FN(xpu_empty_strided));
    m.impl("normal_", TORCH_FN(xpu_normal_));
    m.impl("as_strided", TORCH_FN(xpu_as_strided));
    m.impl("_copy_from", TORCH_FN(xpu_copy_from));
  }

  static std::unique_ptr<torch::detail::TorchLibraryInit> g_xpu_rereg;

  // Exported C function so the test binary can call it without C++ name mangling.
  extern "C" void xpu_reregister_kernels() {
    g_xpu_rereg = std::make_unique<torch::detail::TorchLibraryInit>(
        torch::Library::IMPL,
        &do_register_xpu_kernels,
        "aten",
        std::optional<c10::DispatchKey>(c10::DispatchKey::PrivateUse1),
        __FILE__,
        static_cast<uint32_t>(__LINE__));
  }

}  // namespace xpu
}  // namespace flag_gems

#endif  // FLAGGEMS_USE_KUNLUNXIN
