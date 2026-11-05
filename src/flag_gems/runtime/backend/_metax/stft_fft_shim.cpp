// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <mcfft.h>

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace {
struct Plan {
  mcfftHandle handle {};
};

// These are the bridge's enum values, not assumed mcFFT enum values.
constexpr int kRealToComplex = 0;
constexpr int kComplexToComplex = 1;
constexpr int kForward = -1;
constexpr int kInverse = 1;
constexpr int kInvalidArgument = -1;
constexpr int kAllocationFailed = -2;

int status(mcfftResult result) {
  return result == MCFFT_SUCCESS ? 0 : static_cast<int>(result);
}

template <typename F>
struct StreamArgument;
template <typename R, typename A, typename B>
struct StreamArgument<R (*)(A, B)> {
  using type = B;
};
template <typename R, typename A, typename B>
struct StreamArgument<R (*)(A, B) noexcept> : StreamArgument<R (*)(A, B)> {};

template <typename T>
T stream_from_address(void* address) {
  static_assert(std::is_pointer<T>::value || std::is_integral<T>::value,
                "mcFFT stream must be a pointer or integral handle");
  if constexpr (std::is_pointer<T>::value) {
    return reinterpret_cast<T>(address);
  } else {
    static_assert(sizeof(T) >= sizeof(std::uintptr_t),
                  "mcFFT integral stream handle cannot hold a stream pointer");
    return static_cast<T>(reinterpret_cast<std::uintptr_t>(address));
  }
}

static_assert(sizeof(mcfftReal) == sizeof(float), "mcFFT real storage must match float32");
static_assert(sizeof(mcfftComplex) == 2 * sizeof(float), "mcFFT complex storage must match complex64");
}  // namespace

#define STFT_EXPORT extern "C" __attribute__((visibility("default")))

STFT_EXPORT int gemsMcfftCreate(void** output) {
  if (!output) return kInvalidArgument;
  *output = nullptr;
  auto* plan = new (std::nothrow) Plan;
  if (!plan) return kAllocationFailed;
  const int result = status(mcfftCreate(&plan->handle));
  if (result != 0) {
    delete plan;
    return result;
  }
  *output = plan;
  return 0;
}

STFT_EXPORT int gemsMcfftDestroy(void* opaque) {
  if (!opaque) return kInvalidArgument;
  auto* plan = static_cast<Plan*>(opaque);
  const int result = status(mcfftDestroy(plan->handle));
  if (result == 0) delete plan;
  return result;
}

STFT_EXPORT int gemsMcfftSetAutoAllocation(void* opaque, int enabled) {
  if (!opaque) return kInvalidArgument;
  return status(mcfftSetAutoAllocation(static_cast<Plan*>(opaque)->handle, enabled));
}

STFT_EXPORT int gemsMcfftMakePlanMany(void* opaque,
                                      int rank,
                                      int* n,
                                      int* inembed,
                                      int istride,
                                      int idist,
                                      int* onembed,
                                      int ostride,
                                      int odist,
                                      int transform,
                                      int batch,
                                      std::size_t* work_size) {
  if (!opaque || (transform != kRealToComplex && transform != kComplexToComplex)) return kInvalidArgument;
  const mcfftType type = transform == kRealToComplex ? MCFFT_R2C : MCFFT_C2C;
  return status(mcfftMakePlanMany(static_cast<Plan*>(opaque)->handle,
                                  rank,
                                  n,
                                  inembed,
                                  istride,
                                  idist,
                                  onembed,
                                  ostride,
                                  odist,
                                  type,
                                  batch,
                                  work_size));
}

STFT_EXPORT int gemsMcfftSetStream(void* opaque, void* address) {
  if (!opaque) return kInvalidArgument;
  using Stream = typename StreamArgument<decltype(&mcfftSetStream)>::type;
  return status(mcfftSetStream(static_cast<Plan*>(opaque)->handle, stream_from_address<Stream>(address)));
}

STFT_EXPORT int gemsMcfftSetWorkArea(void* opaque, void* workspace) {
  if (!opaque) return kInvalidArgument;
  return status(mcfftSetWorkArea(static_cast<Plan*>(opaque)->handle, workspace));
}

STFT_EXPORT int gemsMcfftExecR2C(void* opaque, void* input, void* output) {
  if (!opaque) return kInvalidArgument;
  return status(mcfftExecR2C(static_cast<Plan*>(opaque)->handle,
                             static_cast<mcfftReal*>(input),
                             static_cast<mcfftComplex*>(output)));
}

STFT_EXPORT int gemsMcfftExecC2C(void* opaque, void* input, void* output, int direction) {
  if (!opaque || (direction != kForward && direction != kInverse)) return kInvalidArgument;
  const int vendor_direction = direction == kForward ? MCFFT_FORWARD : MCFFT_INVERSE;
  return status(mcfftExecC2C(static_cast<Plan*>(opaque)->handle,
                             static_cast<mcfftComplex*>(input),
                             static_cast<mcfftComplex*>(output),
                             vendor_direction));
}
