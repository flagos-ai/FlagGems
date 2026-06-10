// Phase 2 - Task ctest (α): verify the core equivalence invariant —
//   build_pack(sig, args_with_inline_constexprs...)
//                        == build_pack_with_config(sig, cfg, args_without_constexprs...)
// when `cfg.kwargs` carries the same values as the trailing constexpr block.
//
// This tests the property that lets `autotuned_call` substitute for `operator()`
// without changing the kernel cache key (full_signature). It does NOT exercise
// at::Tensor handling (which would require torch init) — non-constexpr scalars
// + constexpr int + constexpr bool is enough to cover the variant dispatch
// path that B introduces.
//
// Works at the free-function level (build_pack / build_pack_with_config),
// so no TritonJITFunctionImpl construction (which embeds Python) is needed.

#include "triton_jit/config.h"
#include "triton_jit/triton_jit_function.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>

using namespace triton_jit;

static void test_int_and_bool_constexprs() {
  // Shape: [M, N, K, TILE_M, TILE_N, DIVISIBLE_M, IS_FP64]
  //   - 3 NON_CONSTEXPR ints (mirroring runtime dims)
  //   - 2 CONSTEXPR ints (tile sizes)
  //   - 2 CONSTEXPR bools (flags)  ← variant dispatch covers this
  StaticSignature ssig{
      7,
      {
          ArgType::NON_CONSTEXPR,  // M
          ArgType::NON_CONSTEXPR,  // N
          ArgType::NON_CONSTEXPR,  // K
          ArgType::CONSTEXPR,      // TILE_M
          ArgType::CONSTEXPR,      // TILE_N
          ArgType::CONSTEXPR,      // DIVISIBLE_M
          ArgType::CONSTEXPR,      // IS_FP64
      },
  };

  const int M = 256, N = 128, K = 64;

  SignaturePack pack_a = build_pack(
      ssig, M, N, K, int64_t{128}, int64_t{64}, true, false);

  Config cfg{
      /*num_warps=*/4,
      /*num_stages=*/1,
      {
          {"TILE_M", int64_t{128}},
          {"TILE_N", int64_t{64}},
          {"DIVISIBLE_M", true},
          {"IS_FP64", false},
      },
  };
  SignaturePack pack_b = build_pack_with_config(ssig, cfg, M, N, K);

  std::cout << "  pack_a.full_signature = \"" << pack_a.full_signature << "\"\n";
  std::cout << "  pack_b.full_signature = \"" << pack_b.full_signature << "\"\n";

  assert(pack_a.full_signature == pack_b.full_signature);

  // Sanity-check on the expected shape:
  //   - 3 NON_CONSTEXPR ints  → "i32,i32,i32"
  //   - 2 CONSTEXPR int64    → "128,64"
  //   - 2 CONSTEXPR bool     → "true,false" (lowercase per Q1 finding)
  const std::string expected = "i32,i32,i32,128,64,true,false";
  assert(pack_a.full_signature == expected);

  // The non-constexpr args also become buffer entries; constexprs do not.
  // Plus 2 global_scratch slots appended on non-NPU backends.
  // = 3 (non_constexpr) + 2 (global_scratch) = 5 entries.
  assert(pack_a.buffer.size() == 5);
  assert(pack_b.buffer.size() == 5);
}

static void test_empty_kwargs_matches_no_constexpr() {
  // When the kernel has no constexpr block, autotuned_call with empty kwargs
  // must reproduce the same signature as a positional call with no
  // constexpr trailing.
  StaticSignature ssig{
      2,
      {ArgType::NON_CONSTEXPR, ArgType::NON_CONSTEXPR},
  };

  SignaturePack pack_a = build_pack(ssig, int64_t{1024}, int64_t{4});

  Config cfg{/*num_warps=*/8, /*num_stages=*/2, /*kwargs=*/{}};
  SignaturePack pack_b = build_pack_with_config(
      ssig, cfg, int64_t{1024}, int64_t{4});

  assert(pack_a.full_signature == pack_b.full_signature);
  assert(pack_a.full_signature == "i64,i64");
}

int main() {
  std::cout << "[1/2] test_int_and_bool_constexprs\n";
  test_int_and_bool_constexprs();
  std::cout << "[2/2] test_empty_kwargs_matches_no_constexpr\n";
  test_empty_kwargs_matches_no_constexpr();
  std::cout << "signature_equivalence_test: all passed\n";
  return 0;
}
