#include <tvm/ffi/extra/cuda/cubin_launcher.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/tvm_ffi.h>
#include <cstdint>

#ifndef W8A8_BLOCK_FP8_MATMUL_KERNEL_GENERAL_STUB
#define W8A8_BLOCK_FP8_MATMUL_KERNEL_GENERAL_STUB(grid, device, stream, args, kwargs)
#endif

#ifndef W8A8_BLOCK_FP8_MATMUL_GENERAL_TVM_FFI_NAME
#define W8A8_BLOCK_FP8_MATMUL_GENERAL_TVM_FFI_NAME ""
#endif

tvm::ffi::Tensor W8A8BlockFp8MatmulGeneral(tvm::ffi::Tensor a,
                                           tvm::ffi::Tensor b,
                                           tvm::ffi::Tensor c,
                                           tvm::ffi::Tensor as,
                                           tvm::ffi::Tensor bs,
                                           int32_t m,
                                           int32_t n,
                                           int32_t k,
                                           int32_t group_n,
                                           int32_t group_k,
                                           int32_t stride_am,
                                           int32_t stride_ak,
                                           int32_t stride_bk,
                                           int32_t stride_bn,
                                           int32_t stride_cm,
                                           int32_t stride_cn,
                                           int32_t stride_as_m,
                                           int32_t stride_as_k,
                                           int32_t stride_bs_k,
                                           int32_t stride_bs_n) {
  tvm::ffi::Function grid =
      tvm::ffi::Function::FromTyped([m, n](const tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any>& meta)
                                        -> tvm::ffi::Tuple<int32_t, int32_t, int32_t> {
        const int32_t block_m = meta["BLOCK_M"].cast<int32_t>();
        const int32_t block_n = meta["BLOCK_N"].cast<int32_t>();
        const int32_t grid_m = (m + block_m - 1) / block_m;
        const int32_t grid_n = (n + block_n - 1) / block_n;
        return tvm::ffi::Tuple(grid_m * grid_n, 1, 1);
      });

  DLDevice device = a.device();
  void* stream = TVMFFIEnvGetStream(device.device_type, device.device_id);

  tvm::ffi::Array<tvm::ffi::Any> args = {
      a,         b,         c,           as,          bs,          m,           n,
      k,         group_n,   group_k,     stride_am,   stride_ak,   stride_bk,   stride_bn,
      stride_cm, stride_cn, stride_as_m, stride_as_k, stride_bs_k, stride_bs_n,
  };
  tvm::ffi::Map<tvm::ffi::String, tvm::ffi::Any> kwargs = {};

  W8A8_BLOCK_FP8_MATMUL_KERNEL_GENERAL_STUB(grid, device.device_id, stream, args, kwargs);
  return c;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(W8A8_BLOCK_FP8_MATMUL_GENERAL_TVM_FFI_NAME, W8A8BlockFp8MatmulGeneral);
}
