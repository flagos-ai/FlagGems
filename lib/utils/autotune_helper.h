#pragma once
//
// Phase 4 — AutotunedCall: per-kernel mirror cache + bridge to FlagGems
// Python autotune. Sits between `lib/*.cpp` callers and
// `triton_jit::TritonJITFunctionImpl::autotuned_call`.
//
// 4.2 scope (this revision): class skeleton + Q4 TuneKey alignment assertion.
// Mirror cache (4.3), bridge dispatch (4.4), observability (4.5) come in
// subsequent revisions.
//
// Usage shape (from autotune_config_interface_design.md §6.3):
//
//     static AutotunedCall ac("max_kernel.py", "max_kernel",
//                              {"non_reduction_size", "reduction_size"});
//     // ... in op body, eventually:
//     ac.launch(f, raw_stream, gx, gy, gz, {non_red, red}, args...);
//

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "fmt/core.h"
#include "pybind11/pybind11.h"
// Registers `pybind11::detail::type_caster<at::Tensor>` so py::cast /
// py::make_tuple can convert ATen tensors to Python torch.Tensor objects
// (defined in libtorch_python; the operators target must also link it).
#include "torch/csrc/utils/pybind.h"

#include "triton_jit/config.h"
#include "triton_jit/jit_utils.h"

namespace flag_gems {

// V0 TuneKey: positional vector of int64. Covers bmm / baddbmm / max
// (their `key=[...]` only references int-valued args). float / str keys
// reserved for V1.
using TuneKey = std::vector<int64_t>;

// Look up an int-valued constexpr by name in a Config's kwargs. Throws
// if the name is not present or its variant slot holds a bool. Used by
// callers that need a tuned constexpr value to compute a grid dim (e.g.
// `num_blocks = cdiv(M, BLOCK_M)`).
inline int64_t get_int_kwarg(const triton_jit::Config& cfg, std::string_view name) {
  for (const auto& [k, v] : cfg.kwargs) {
    if (k == name) {
      if (std::holds_alternative<int64_t>(v)) {
        return std::get<int64_t>(v);
      }
      throw std::runtime_error(fmt::format("get_int_kwarg: '{}' is bool-typed, not int", name));
    }
  }
  throw std::runtime_error(fmt::format("get_int_kwarg: no kwarg named '{}' in Config", name));
}

struct TuneKeyHash {
  size_t operator()(const TuneKey& k) const noexcept {
    // boost::hash_combine-style; adequate for cache lookup, not adversarial.
    size_t h = k.size();
    for (int64_t x : k) {
      h ^= std::hash<int64_t>()(x) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

// Owns the mirror cache (4.3+) and the bridge dispatch for one kernel.
// One static instance per kernel call site in lib/*.cpp.
class AutotunedCall {
 public:
  // expected_key_names must match — in the same order — the Python LibTuner's
  // `key=[...]` list. Asserted lazily on first launch (or via validate()).
  AutotunedCall(std::string_view kernel_path,
                std::string_view function_name,
                std::vector<std::string> expected_key_names)
      : kernel_path_(kernel_path),
        function_name_(function_name),
        expected_key_names_(std::move(expected_key_names)) {
  }

  AutotunedCall(const AutotunedCall&) = delete;
  AutotunedCall& operator=(const AutotunedCall&) = delete;
  AutotunedCall(AutotunedCall&&) = delete;
  AutotunedCall& operator=(AutotunedCall&&) = delete;

  // Cross-language alignment check (Q4 in progress.md). Idempotent — only
  // the first call's bridge query runs; subsequent calls are no-ops. Throws
  // std::runtime_error on mismatch with a message naming the kernel,
  // position, and both sides' names.
  //
  // Requires Python to already be initialized; this is the case whenever a
  // TritonJITFunctionImpl has been constructed in the program, which
  // happens for every FlagGems lib op. AutotunedCall ctor itself does no
  // Python work (so static-init-order is safe).
  void validate() {
    std::call_once(validate_once_, [this]() { do_validate_(); });
  }

  const std::string& kernel_path() const noexcept {
    return kernel_path_;
  }
  const std::string& function_name() const noexcept {
    return function_name_;
  }
  const std::vector<std::string>& expected_key_names() const noexcept {
    return expected_key_names_;
  }

  // Mirror cache primitive (4.3). Look up `key`; on miss, call `computer(key)`
  // to produce a Config, insert it, and return a reference to the stored
  // value. The reference remains valid for the lifetime of *this — V0 does
  // not evict from the cache, and unordered_map's references are stable
  // across rehashes (only iterators are invalidated).
  //
  // `computer` runs while holding the relevant stripe lock. If it throws,
  // the exception propagates out and the key is NOT cached.
  //
  // Thread-safe under concurrent calls to distinct stripes (1/64 the
  // contention of a global lock). Same-key concurrent callers serialize on
  // the stripe lock; one wins the compute, others observe the cached value
  // on their second look-up after acquiring the lock.
  //
  // Does NOT auto-invoke validate() — bridge dispatch (4.4) wires that up.
  template <typename ComputerFn>
  const triton_jit::Config& lookup_or_compute(const TuneKey& key, ComputerFn&& computer) {
    Stripe& s = stripes_[bucket_index(key)];
    std::lock_guard<std::mutex> lock(s.mu);

    auto it = s.map.find(key);
    if (it != s.map.end()) {
      return it->second;
    }

    triton_jit::Config cfg = computer(key);
    auto [new_it, inserted] = s.map.emplace(key, std::move(cfg));
    if (inserted) {
      size_t new_total = total_size_.fetch_add(1, std::memory_order_relaxed) + 1;
      if (new_total > kCapWarning && !cap_warning_logged_.test_and_set()) {
        std::cerr << "[autotune_helper] WARNING: mirror cache size " << new_total << " > " << kCapWarning
                  << " for " << function_name_ << " (V0 has no eviction; expect monotonic growth)\n";
      }
    }
    return new_it->second;
  }

  // Total entries cached across all stripes. Useful for tests.
  size_t cache_size() const noexcept {
    return total_size_.load(std::memory_order_relaxed);
  }

  // Top-level entry: validate alignment (lazy fail-fast), look up in mirror
  // cache, dispatch to Python bridge on miss. Returns a stable reference to
  // the cached Config.
  //
  // `grid_fn` is invoked with a Config snapshot if the LibTuner SQL cache
  // misses and Triton autotuner has to benchmark candidate configs (see
  // Q12 in progress.md). For SQL-cached shapes (the common case) grid_fn
  // is never called. Signature:
  //
  //     std::tuple<unsigned, unsigned, unsigned> grid_fn(const Config& cfg)
  //
  // The Config passed to grid_fn aggregates the kernel's named args
  // (M, N, ...) and the candidate's autotune constexprs (BLOCK_M, ...) as
  // ConfigValue entries -- use get_int_kwarg to extract any of them.
  //
  // `args` is the kernel's positional non-constexpr prefix (same shape that
  // `autotuned_call(stream, gx, gy, gz, cfg, args...)` takes). pybind11
  // py::cast handles each element; at::Tensor cast requires PyTorch's
  // bindings (`torch/csrc/utils/pybind.h` header + libtorch_python link).
  template <typename GridFn, typename... Args>
  const triton_jit::Config& lookup(const TuneKey& named_key, GridFn grid_fn, Args... args) {
    validate();
    // dtype-aware effective key: named keys (M,N,K,...) + each tensor arg's
    // dtype, mirroring LibTuner.get_key. Closes the cross-dtype reuse gap.
    TuneKey key = named_key;
    const size_t dtype_count = append_arg_dtypes_(key, args...);
    return lookup_or_compute(key, [this, &args..., &key, &grid_fn, dtype_count](const TuneKey&) {
      triton_jit::Config cfg = dispatch_to_bridge_(grid_fn, dtype_count, args...);
      if (log_enabled_()) {
        log_miss_(key, cfg);
      }
      return cfg;
    });
  }

  static constexpr size_t kNumStripes = 64;
  static constexpr size_t kCapWarning = 1024;

 private:
  struct Stripe {
    std::mutex mu;
    std::unordered_map<TuneKey, triton_jit::Config, TuneKeyHash> map;
  };

  size_t bucket_index(const TuneKey& key) const noexcept {
    return TuneKeyHash {}(key) % kNumStripes;
  }

  // Append each at::Tensor arg's dtype (scalar_type as int64) to `key`,
  // mirroring Python LibTuner.get_key:
  //   key += tuple(str(a.dtype) for a in args.values() if hasattr(a,"dtype"))
  // Non-tensor args are skipped at compile time. Returns the count appended,
  // used to assert C++/Python agree on dtype-key composition.
  template <typename... Args>
  static size_t append_arg_dtypes_(TuneKey& key, const Args&... args) {
    size_t n = 0;
    auto one = [&](const auto& a) {
      using T = std::decay_t<decltype(a)>;
      if constexpr (std::is_same_v<T, at::Tensor>) {
        key.push_back(static_cast<int64_t>(a.scalar_type()));
        ++n;
      }
    };
    (one(args), ...);
    return n;
  }

  // Call the Python bridge to resolve a Config. Acquires GIL. Converts
  // C++ args to a py::tuple (relying on pybind11's per-element py::cast),
  // calls autotune_bridge.lookup_config(...), and unpacks the returned
  // dict into a triton_jit::Config.
  //
  // py::error_already_set is caught and rethrown as std::runtime_error
  // tagged with the kernel name so callers see who failed without
  // unwinding pybind11 internals.
  template <typename GridFn, typename... Args>
  triton_jit::Config dispatch_to_bridge_(GridFn& grid_fn, size_t cpp_dtype_count, Args&... args) {
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    // Ensure the bridge module is importable (script_dir on sys.path).
    // do_validate_ already did this on first launch, but lookup() can also
    // be called from a TU that bypasses validate() in tests; replay the
    // path insert for idempotency.
    std::filesystem::path script_dir = triton_jit::get_script_dir();
    py::module_::import("sys").attr("path").attr("insert")(0, script_dir.string());

    // Wrap C++ grid_fn into a Python callable for the Triton autotuner's
    // bench loop (Q12). Triton calls grid(meta) where meta is a dict of
    // {arg_name: value, **config.all_kwargs()} -- the union of kernel args
    // and the candidate's tuned constexprs. We synthesize a Config from
    // the int/bool entries of meta and hand it to the C++ grid_fn.
    py::function py_grid = py::cpp_function([grid_fn](py::dict meta) -> py::tuple {
      triton_jit::Config tmp_cfg;
      for (auto item : meta) {
        std::string k = py::str(item.first).cast<std::string>();
        // num_warps/num_stages are launch params, not constexpr block
        if (k == "num_warps" || k == "num_stages") continue;
        py::object v = py::reinterpret_borrow<py::object>(item.second);
        // bool MUST be checked before int (Python bool is int subclass)
        if (py::isinstance<py::bool_>(v)) {
          tmp_cfg.kwargs.emplace_back(std::move(k), triton_jit::ConfigValue {v.cast<bool>()});
        } else if (py::isinstance<py::int_>(v)) {
          tmp_cfg.kwargs.emplace_back(std::move(k), triton_jit::ConfigValue {v.cast<int64_t>()});
        }
        // Skip tensors / floats / strings -- not in V0 Config.
      }
      auto [gx, gy, gz] = grid_fn(tmp_cfg);
      return py::make_tuple(gx, gy, gz);
    });

    py::dict result;
    try {
      py::tuple py_args = py::make_tuple(args...);
      py::dict py_kwargs;
      // Plumbing for Q12: Triton autotuner's `_bench` forwards **meta to
      // `JITFunction.run(*args, **current)`; without grid/warmup it errors
      // with "missing 2 required keyword-only arguments".
      py_kwargs["grid"] = py_grid;
      py_kwargs["warmup"] = py::bool_(false);
      py::module_ bridge = py::module_::import("autotune_bridge");
      result =
          bridge.attr("lookup_config")(kernel_path_, function_name_, py_args, py_kwargs).cast<py::dict>();
    } catch (const py::error_already_set& e) {
      throw std::runtime_error(fmt::format("autotune_bridge.lookup_config failed for {}::{}: {}",
                                           kernel_path_,
                                           function_name_,
                                           e.what()));
    }

    triton_jit::Config cfg;
    cfg.num_warps = result["num_warps"].cast<int>();
    cfg.num_stages = result["num_stages"].cast<int>();

    py::list py_kwargs_list = result["kwargs"].cast<py::list>();
    cfg.kwargs.reserve(py::len(py_kwargs_list));
    for (auto item : py_kwargs_list) {
      py::tuple pair = item.cast<py::tuple>();
      std::string name = pair[0].cast<std::string>();
      py::object value = pair[1];
      // bool MUST be checked before int -- py::bool_ values would also
      // succeed as py::int_ but the variant slot semantics differ.
      if (py::isinstance<py::bool_>(value)) {
        cfg.kwargs.emplace_back(std::move(name), triton_jit::ConfigValue {value.cast<bool>()});
      } else {
        cfg.kwargs.emplace_back(std::move(name), triton_jit::ConfigValue {value.cast<int64_t>()});
      }
    }

    // Cross-language guard (extends Q4): C++ appended cpp_dtype_count tensor
    // dtypes to the mirror key; Python get_key appends key_dtype_count. They
    // must match or the two caches partition by dtype differently.
    if (result.contains("key_dtype_count")) {
      const size_t py_dtype_count = result["key_dtype_count"].cast<size_t>();
      std::call_once(dtype_check_once_, [&]() {
        if (py_dtype_count != cpp_dtype_count) {
          throw std::runtime_error(
              fmt::format("AutotunedCall: dtype-key mismatch for {}::{} -- C++ appended {} "
                          "tensor dtype(s), Python get_key appends {}. Mirror cache and "
                          "LibTuner cache would partition by dtype differently.",
                          kernel_path_,
                          function_name_,
                          cpp_dtype_count,
                          py_dtype_count));
        }
      });
    }
    return cfg;
  }

  // 4.5 observability. Env-gated; uncached so tests can flip across cases
  // without spawning new processes. Per-dispatch getenv cost is dwarfed by
  // the bridge call (~940 µs/miss measured in Q3).
  static bool log_enabled_() noexcept {
    const char* env = std::getenv("LIBTRITON_JIT_LOG_AUTOTUNE");
    return env != nullptr && std::strcmp(env, "1") == 0;
  }

  void log_miss_(const TuneKey& key, const triton_jit::Config& cfg) const {
    std::cerr << "[autotune] miss " << function_name_ << " key=(";
    for (size_t i = 0; i < key.size(); ++i) {
      if (i > 0) std::cerr << ",";
      const std::string& name = (i < expected_key_names_.size()) ? expected_key_names_[i] : std::string {"?"};
      std::cerr << name << "=" << key[i];
    }
    std::cerr << ") -> num_warps=" << cfg.num_warps << " num_stages=" << cfg.num_stages << " kwargs=[";
    for (size_t i = 0; i < cfg.kwargs.size(); ++i) {
      if (i > 0) std::cerr << ",";
      std::cerr << cfg.kwargs[i].first << "=";
      std::visit(
          [](auto&& v) {
            using T = std::decay_t<decltype(v)>;
            // bool: emit "true"/"false" for grep parity with Python str();
            // default `operator<<` would print "1"/"0".
            if constexpr (std::is_same_v<T, bool>) {
              std::cerr << (v ? "true" : "false");
            } else {
              std::cerr << v;
            }
          },
          cfg.kwargs[i].second);
    }
    std::cerr << "]\n";
  }

  void do_validate_() {
    namespace py = pybind11;
    py::gil_scoped_acquire gil;

    // The bridge script lives in libtriton_jit's scripts/ dir; ensure it's
    // on sys.path before importing (same pattern triton_jit_function.cpp uses).
    std::filesystem::path script_dir = triton_jit::get_script_dir();
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.string());

    py::list py_names;
    try {
      py::module_ bridge = py::module_::import("autotune_bridge");
      py_names = bridge.attr("get_tune_key_names")(kernel_path_, function_name_).cast<py::list>();
    } catch (const py::error_already_set& e) {
      throw std::runtime_error(fmt::format("autotune_bridge.get_tune_key_names failed for {}::{}: {}",
                                           kernel_path_,
                                           function_name_,
                                           e.what()));
    }

    if (py_names.size() != expected_key_names_.size()) {
      throw std::runtime_error(
          fmt::format("AutotunedCall: TuneKey arity mismatch for {}::{} -- "
                      "C++ caller declared {} key names, Python LibTuner has {}. "
                      "Update the AutotunedCall constructor's expected_key_names to "
                      "match the Python @libtuner key=[...] list exactly.",
                      kernel_path_,
                      function_name_,
                      expected_key_names_.size(),
                      py_names.size()));
    }

    for (size_t i = 0; i < py_names.size(); ++i) {
      std::string py_name = py_names[i].cast<std::string>();
      if (py_name != expected_key_names_[i]) {
        throw std::runtime_error(
            fmt::format("AutotunedCall: TuneKey name mismatch for {}::{} at position {} -- "
                        "C++ expected \"{}\", Python LibTuner has \"{}\". Reorder the "
                        "AutotunedCall constructor's expected_key_names to match the "
                        "Python @libtuner key=[...] list exactly.",
                        kernel_path_,
                        function_name_,
                        i,
                        expected_key_names_[i],
                        py_name));
      }
    }
  }

  std::string kernel_path_;
  std::string function_name_;
  std::vector<std::string> expected_key_names_;
  std::once_flag validate_once_;
  std::once_flag dtype_check_once_;

  std::array<Stripe, kNumStripes> stripes_;
  std::atomic<size_t> total_size_ {0};
  std::atomic_flag cap_warning_logged_ {};  // C++20 default-init: clear
};

}  // namespace flag_gems
