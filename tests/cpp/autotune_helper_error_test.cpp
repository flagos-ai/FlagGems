// Phase 4.6 (b): error propagation from the Python bridge across the
// pybind11 boundary into C++ runtime_error, including:
//   - validate() wrapping  (bridge raises during get_tune_key_names)
//   - dispatch_to_bridge_ wrapping  (bridge raises during lookup_config)
//   - cache invariants on bridge error (no caching, recovery succeeds)
//   - std::call_once retry semantics on validate failure
//
// Trigger mechanisms:
//   - validate error:   non-existent kernel_path -> Python ImportError /
//                       FileNotFoundError surfaces through the bridge's
//                       `_load_kernel` -> RuntimeError
//   - dispatch error:   call lookup() with too few positional args
//                       (only A, B; M and N missing) -> the synth_kernel
//                       heuristic's `nargs["M"]` raises KeyError

#include "utils/autotune_helper.h"

#include <pybind11/embed.h>

#include <cassert>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace fg = flag_gems;

static const char* kKernelPath =
    "/workspace/FlagGems/tests/cpp/autotune_bridge_synth_kernel.py";

static void case_validate_error_wraps_bridge_failure() {
  // Path that exists nowhere. _load_kernel inside the bridge will raise.
  fg::AutotunedCall ac(
      "/definitely/not/a/real/path/synthetic.py",
      "synth_kernel",
      {"M", "N"});

  try {
    ac.validate();
  } catch (const std::runtime_error& e) {
    std::string what = e.what();
    // Wrapped, kernel name included, includes the Python-side message
    assert(what.find("autotune_bridge.get_tune_key_names failed") != std::string::npos);
    assert(what.find("synth_kernel") != std::string::npos);
    assert(what.find("/definitely/not/a/real/path/synthetic.py") != std::string::npos);
    // spec_from_file_location succeeds for nonexistent files, but exec_module
    // fails with FileNotFoundError surfaced through importlib.
    assert(what.find("FileNotFoundError") != std::string::npos);
    return;
  }
  throw std::runtime_error("case_validate_error_wraps_bridge_failure: expected throw");
}

static void case_validate_retries_on_failure() {
  // std::call_once does NOT mark the flag as called when the callable
  // throws -- a subsequent validate() must re-attempt (and re-throw).
  fg::AutotunedCall ac(
      "/definitely/not/a/real/path/synthetic.py",
      "synth_kernel",
      {"M", "N"});

  bool first_threw = false;
  try { ac.validate(); } catch (const std::runtime_error&) { first_threw = true; }
  assert(first_threw);

  bool second_threw = false;
  try { ac.validate(); } catch (const std::runtime_error&) { second_threw = true; }
  assert(second_threw);
}

static void case_dispatch_error_wraps_and_does_not_cache() {
  // validate() succeeds (synth_kernel is real); dispatch fails because the
  // heuristic needs M/N in the named-args dict but we pass only A, B.
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});
  assert(ac.cache_size() == 0);

  auto dummy_grid = [](const triton_jit::Config&)
      -> std::tuple<unsigned, unsigned, unsigned> { return {1u, 1u, 1u}; };

  fg::TuneKey key = {256, 128};
  try {
    // Pass only 2 args (A, B). zip(arg_names, args) gives {"A": ..., "B": ...},
    // missing "M"/"N" -> heuristic's nargs["M"] raises KeyError.
    ac.lookup(key, dummy_grid, int64_t{0}, int64_t{0});
  } catch (const std::runtime_error& e) {
    std::string what = e.what();
    assert(what.find("autotune_bridge.lookup_config failed") != std::string::npos);
    assert(what.find("synth_kernel") != std::string::npos);
    assert(what.find("KeyError") != std::string::npos);
    assert(what.find("'M'") != std::string::npos);
    // Crucial: nothing got cached on the failed dispatch.
    assert(ac.cache_size() == 0);
    return;
  }
  throw std::runtime_error("case_dispatch_error_wraps_and_does_not_cache: expected throw");
}

static void case_dispatch_error_then_recovers() {
  fg::AutotunedCall ac(kKernelPath, "synth_kernel", {"M", "N"});
  auto dummy_grid = [](const triton_jit::Config&)
      -> std::tuple<unsigned, unsigned, unsigned> { return {1u, 1u, 1u}; };

  // First call: bad args -> throws, nothing cached
  bool threw = false;
  try {
    ac.lookup(fg::TuneKey{256, 128}, dummy_grid, int64_t{0}, int64_t{0});
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(ac.cache_size() == 0);

  // Second call: correct args -> succeeds, cache populated
  const auto& cfg = ac.lookup(
      fg::TuneKey{256, 128}, dummy_grid,
      int64_t{0}, int64_t{0}, int64_t{256}, int64_t{128});
  assert(cfg.num_warps == 4);
  assert(ac.cache_size() == 1);
}

int main() {
  py::scoped_interpreter guard{};

  std::cout << "[1/4] validate error wraps py::error_already_set\n";
  case_validate_error_wraps_bridge_failure();
  std::cout << "[2/4] validate retries on failure (call_once doesn't latch on throw)\n";
  case_validate_retries_on_failure();
  std::cout << "[3/4] dispatch error wraps + no caching\n";
  case_dispatch_error_wraps_and_does_not_cache();
  std::cout << "[4/4] dispatch error is recoverable on next call\n";
  case_dispatch_error_then_recovers();
  std::cout << "autotune_helper_error_test: all 4 passed\n";
  return 0;
}
