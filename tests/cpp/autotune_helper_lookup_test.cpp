// Phase 4.4 runtime test: end-to-end AutotunedCall::lookup through the
// Python bridge, against the synthetic kernel. No GPU needed (single-config
// LibTuner short-circuits resolve_config; heuristics are pure Python).
//
// Three cases:
//   1. miss -> Config returned with expected num_warps/num_stages and
//      kwargs in the exact kernel positional order, types preserved
//   2. same key again -> cache hit; same stable reference, cache_size
//      stays at 1 (verifies dispatch did NOT re-run)
//   3. new key (different M flips heuristic DIVISIBLE_M to false) ->
//      cache miss, cache_size grows to 2, returned Config has the
//      heuristic-flipped value

#include "utils/autotune_helper.h"

#include <pybind11/embed.h>

#include <cassert>
#include <iostream>
#include <string>
#include <variant>
#include <vector>

namespace py = pybind11;
namespace fg = flag_gems;

static const char* kKernelPath = "/workspace/FlagGems/tests/cpp/autotune_bridge_synth_kernel.py";

// Dummy grid lambda for tests that don't exercise the bench path (synth
// kernel has a single LibTuner config, so resolve_config short-circuits).
static auto dummy_grid_fn = [](const triton_jit::Config&) -> std::tuple<unsigned, unsigned, unsigned> {
  return {1u, 1u, 1u};
};

static const triton_jit::Config& lookup_for_m_n(fg::AutotunedCall& ac, int64_t M, int64_t N) {
  // The synth_kernel's non-constexpr prefix is (A, B, M, N). We pass int
  // placeholders for A, B since the kernel is never executed and heuristics
  // only read "M" and "N" out of the named-args dict.
  return ac.lookup(fg::TuneKey {M, N}, dummy_grid_fn, int64_t {0}, int64_t {0}, M, N);
}

static void case_miss_returns_expected_config() {
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  const triton_jit::Config& cfg = lookup_for_m_n(ac, 256, 128);

  assert(cfg.num_warps == 4);
  assert(cfg.num_stages == 2);
  assert(cfg.kwargs.size() == 6);

  // Kernel positional order: TILE_M, TILE_N, GROUP_M, DIVISIBLE_M,
  // DIVISIBLE_N, IS_FP64
  assert(cfg.kwargs[0].first == "TILE_M");
  assert(cfg.kwargs[1].first == "TILE_N");
  assert(cfg.kwargs[2].first == "GROUP_M");
  assert(cfg.kwargs[3].first == "DIVISIBLE_M");
  assert(cfg.kwargs[4].first == "DIVISIBLE_N");
  assert(cfg.kwargs[5].first == "IS_FP64");

  // Type discrimination: TILE_* / GROUP_M are int64; DIVISIBLE_* / IS_FP64
  // are bool. Variant index 0 = int64_t, 1 = bool.
  assert(cfg.kwargs[0].second.index() == 0);  // int64
  assert(cfg.kwargs[3].second.index() == 1);  // bool
  assert(cfg.kwargs[5].second.index() == 1);  // bool

  assert(std::get<int64_t>(cfg.kwargs[0].second) == 128);
  assert(std::get<int64_t>(cfg.kwargs[1].second) == 64);
  assert(std::get<int64_t>(cfg.kwargs[2].second) == 8);
  assert(std::get<bool>(cfg.kwargs[3].second) == true);  // 256 % 128 == 0
  assert(std::get<bool>(cfg.kwargs[4].second) == true);  // 128 % 64  == 0
  assert(std::get<bool>(cfg.kwargs[5].second) == false);

  assert(ac.cache_size() == 1);
}

static void case_same_key_is_cache_hit() {
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  const triton_jit::Config& first = lookup_for_m_n(ac, 256, 128);
  size_t after_first = ac.cache_size();
  assert(after_first == 1);

  const triton_jit::Config& second = lookup_for_m_n(ac, 256, 128);
  assert(&first == &second);               // stable reference -> same slot
  assert(ac.cache_size() == after_first);  // no new entry
}

static void case_new_key_re_dispatches_and_heuristic_flips() {
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});

  const triton_jit::Config& cfg1 = lookup_for_m_n(ac, 256, 128);
  assert(std::get<bool>(cfg1.kwargs[3].second) == true);  // DIVISIBLE_M
  assert(ac.cache_size() == 1);

  // 200 % 128 != 0 -> heuristic returns False; only possible if dispatch
  // actually re-ran for this new key
  const triton_jit::Config& cfg2 = lookup_for_m_n(ac, 200, 128);
  assert(std::get<bool>(cfg2.kwargs[3].second) == false);
  assert(ac.cache_size() == 2);
  assert(&cfg1 != &cfg2);  // distinct cache slots
}

int main() {
  py::scoped_interpreter guard {};

  std::cout << "[1/3] miss returns expected Config (full structure + types)\n";
  case_miss_returns_expected_config();
  std::cout << "[2/3] same key is cache hit (stable ref, size unchanged)\n";
  case_same_key_is_cache_hit();
  std::cout << "[3/3] new key re-dispatches; heuristic flips DIVISIBLE_M\n";
  case_new_key_re_dispatches_and_heuristic_flips();
  std::cout << "autotune_helper_lookup_test: all 3 passed\n";
  return 0;
}
