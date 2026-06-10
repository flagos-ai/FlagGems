// Phase 2 - Task B compile probe: forces template instantiation of
// `TritonJITFunctionImpl<CudaBackend>::autotuned_call<...>` so the compiler
// type-checks the method body. Class-template explicit instantiation
// (`template class TritonJITFunctionImpl<CudaBackend>;` in triton_jit_function.cpp)
// instantiates regular members but NOT template member functions — those need
// a real call site. This probe is that call site.
//
// We never actually launch a kernel; the call is guarded by `if (false)` so
// no GPU / no live function registration is required. Compiling this TU to .o
// is sufficient to validate body correctness.

#include "triton_jit/backends/cuda_backend.h"
#include "triton_jit/triton_jit_function.h"

#include <cstdint>

namespace {

using F = triton_jit::TritonJITFunctionImpl<triton_jit::CudaBackend>;

// Distinct overloads to exercise both the all-positional path and the
// kwargs-injection path of autotuned_call.
void force_instantiate_empty_kwargs(F* f, triton_jit::CudaBackend::StreamType s) {
  triton_jit::Config cfg{/*num_warps=*/4, /*num_stages=*/1, /*kwargs=*/{}};
  if (f != nullptr) {
    f->autotuned_call(s, 1u, 1u, 1u, cfg, int32_t{0}, float{0.f});
  }
}

void force_instantiate_mixed_kwargs(F* f, triton_jit::CudaBackend::StreamType s) {
  triton_jit::Config cfg{
      /*num_warps=*/8,
      /*num_stages=*/2,
      {
          {"TILE_M", int64_t{128}},
          {"TILE_N", int64_t{64}},
          {"DIVISIBLE_M", true},
          {"IS_FP64", false},
      },
  };
  if (f != nullptr) {
    f->autotuned_call(s, 64u, 1u, 1u, cfg, int32_t{0}, int64_t{0});
  }
}

}  // namespace
