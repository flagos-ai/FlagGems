// Phase 2 - Task A probe: verify libtriton_jit's Config POD definition.
//
// Goals:
//   1. Aggregate initialization works in the forms shown in the design doc
//      (autotune_config_interface_design.md §4.2 and §8 W1).
//   2. variant<int64_t, bool> disambiguates correctly for int literals
//      vs `true`/`false`, without requiring `std::in_place_type` ceremony
//      at the call site (`bmm.cpp` will write Config literals inline).
//   3. The header is self-contained (no transitive include needed).
//
// This is a compile + assertion check, not a behavioural test. The signature
// stringification & autotuned_call wiring are exercised in step B/C of Phase 2.

#include "triton_jit/config.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <variant>

int main() {
  using triton_jit::Config;
  using triton_jit::ConfigValue;

  Config cfg{
      /*num_warps=*/4,
      /*num_stages=*/1,
      {
          {"TILE_M", int64_t{128}},
          {"TILE_N", int64_t{64}},
          {"GROUP_M", int64_t{8}},
          {"DIVISIBLE_M", true},
          {"DIVISIBLE_N", false},
          {"IS_FP64", false},
      },
  };

  assert(cfg.num_warps == 4);
  assert(cfg.num_stages == 1);
  assert(cfg.kwargs.size() == 6);

  // Insertion order preserved — required because kwargs are forwarded to
  // the kernel's constexpr params positionally.
  assert(cfg.kwargs[0].first == "TILE_M");
  assert(cfg.kwargs[5].first == "IS_FP64");

  // variant disambiguation: int literal → int64_t slot (index 0);
  //                        bool literal → bool slot (index 1).
  // If this were ambiguous, brace-init would refuse to compile due to
  // narrowing in the unselected alternative.
  assert(cfg.kwargs[0].second.index() == 0);  // TILE_M -> int64_t
  assert(cfg.kwargs[3].second.index() == 1);  // DIVISIBLE_M -> bool
  assert(cfg.kwargs[5].second.index() == 1);  // IS_FP64 -> bool

  assert(std::get<int64_t>(cfg.kwargs[0].second) == 128);
  assert(std::get<bool>(cfg.kwargs[3].second) == true);
  assert(std::get<bool>(cfg.kwargs[5].second) == false);

  // Empty kwargs is also valid (e.g. a kernel with only num_warps/num_stages
  // tuned and no constexpr block params).
  Config bare{/*num_warps=*/8, /*num_stages=*/2, {}};
  assert(bare.kwargs.empty());

  std::cout << "config_construct_test: all assertions passed\n";
  return 0;
}
