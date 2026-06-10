// Phase 4.3 runtime test: exercise flag_gems::AutotunedCall::lookup_or_compute
// purely at the cache layer — no Python interaction (validate() is NOT
// called; bridge dispatch comes in 4.4).
//
// Five cases:
//   1. miss -> computer called once, value returned and cached
//   2. hit  -> computer NOT called on second lookup; identical reference
//   3. distinct keys (potentially same stripe) -> computer called per key
//   4. computer throws -> exception propagates, key not cached, total_size
//                          unchanged
//   5. capacity warning -> single stderr line at first crossing of 1024

#include "utils/autotune_helper.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fg = flag_gems;

static int compute_count = 0;
static triton_jit::Config make_cfg(int num_warps) {
  return triton_jit::Config {num_warps, 1, {}};
}

static void case_miss_then_hit() {
  fg::AutotunedCall ac("/dev/null", "k", {"M", "N"});
  compute_count = 0;

  fg::TuneKey k = {256, 128};
  const triton_jit::Config& c1 = ac.lookup_or_compute(k, [](const fg::TuneKey&) {
    ++compute_count;
    return make_cfg(8);
  });
  assert(compute_count == 1);
  assert(c1.num_warps == 8);
  assert(ac.cache_size() == 1);

  const triton_jit::Config& c2 = ac.lookup_or_compute(k, [](const fg::TuneKey&) {
    ++compute_count;
    return make_cfg(99);  // sentinel that must NOT win
  });
  assert(compute_count == 1);  // unchanged
  assert(c2.num_warps == 8);
  assert(&c1 == &c2);  // same slot in the map => same reference
  assert(ac.cache_size() == 1);
}

static void case_distinct_keys() {
  fg::AutotunedCall ac("/dev/null", "k", {"M", "N"});
  compute_count = 0;

  // 200 distinct keys, deliberately small spread so multiple land in the
  // same stripe (verifies stripe lock doesn't conflate keys).
  for (int i = 0; i < 200; ++i) {
    fg::TuneKey k = {i, i + 1};
    const auto& cfg = ac.lookup_or_compute(k, [i](const fg::TuneKey&) {
      ++compute_count;
      return make_cfg(i + 1);
    });
    assert(cfg.num_warps == i + 1);
  }
  assert(compute_count == 200);
  assert(ac.cache_size() == 200);

  // Re-fetch all 200: zero new computes
  int reused = compute_count;
  for (int i = 0; i < 200; ++i) {
    fg::TuneKey k = {i, i + 1};
    const auto& cfg = ac.lookup_or_compute(k, [](const fg::TuneKey&) {
      ++compute_count;
      return make_cfg(0);
    });
    (void)cfg;
  }
  assert(compute_count == reused);
}

static void case_computer_throws() {
  fg::AutotunedCall ac("/dev/null", "k", {"M", "N"});

  fg::TuneKey k = {7, 7};
  try {
    ac.lookup_or_compute(k, [](const fg::TuneKey&) -> triton_jit::Config {
      throw std::runtime_error("compute failed");
    });
  } catch (const std::runtime_error& e) {
    assert(std::string(e.what()) == "compute failed");
    assert(ac.cache_size() == 0);  // nothing cached on throw

    // Subsequent successful lookup with the same key should compute fresh.
    int called = 0;
    const auto& cfg = ac.lookup_or_compute(k, [&called](const fg::TuneKey&) {
      ++called;
      return make_cfg(4);
    });
    assert(called == 1);
    assert(cfg.num_warps == 4);
    assert(ac.cache_size() == 1);
    return;
  }
  throw std::runtime_error("case_computer_throws: exception did not propagate");
}

static void case_capacity_warning() {
  // Redirect stderr to a stringstream so we can assert on the warning.
  std::stringstream captured;
  std::streambuf* old_cerr = std::cerr.rdbuf(captured.rdbuf());

  fg::AutotunedCall ac("/dev/null", "k", {"M", "N"});
  // Insert 1025 entries — first 1024 silent, the 1025th must emit one warning.
  for (int i = 0; i < 1025; ++i) {
    fg::TuneKey k = {i, 0};
    ac.lookup_or_compute(k, [](const fg::TuneKey&) { return make_cfg(4); });
  }
  // Insert more — must NOT emit additional warnings (single-shot flag).
  for (int i = 1025; i < 1050; ++i) {
    fg::TuneKey k = {i, 0};
    ac.lookup_or_compute(k, [](const fg::TuneKey&) { return make_cfg(4); });
  }

  std::cerr.rdbuf(old_cerr);  // restore

  std::string log = captured.str();
  assert(log.find("mirror cache size") != std::string::npos);
  assert(log.find("1025") != std::string::npos);
  assert(log.find("> 1024") != std::string::npos);

  // Exactly ONE warning line:
  size_t count = 0;
  size_t pos = 0;
  while ((pos = log.find("WARNING", pos)) != std::string::npos) {
    ++count;
    pos += 1;
  }
  assert(count == 1);

  assert(ac.cache_size() == 1050);
}

int main() {
  std::cout << "[1/4] miss-then-hit, identical reference\n";
  case_miss_then_hit();
  std::cout << "[2/4] distinct keys: each computed once, all hits on second pass\n";
  case_distinct_keys();
  std::cout << "[3/4] computer throws: propagates, no caching\n";
  case_computer_throws();
  std::cout << "[4/4] capacity warning: single-shot at 1024 crossing\n";
  case_capacity_warning();
  std::cout << "autotune_helper_cache_test: all 4 passed\n";
  return 0;
}
