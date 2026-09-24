"""Test-local shape and fresh-input support for the generated operator suites.

Candidate dispatch, Preflight reports and profiling hooks belong to upstream
benchmark.base. This helper does not add another candidate registry.
"""
import time
from contextlib import nullcontext

import torch
import yaml

from . import base, consts

def _clone_benchmark_inputs(value):
    # The opt-in cases use independent dense/sparse tensors and scalar kwargs.
    # Clone before each invocation, outside timing/profiler boundaries.
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_benchmark_inputs(x) for x in value)
    if isinstance(value, dict):
        return {k: _clone_benchmark_inputs(v) for k, v in value.items()}
    return value


class OperatorBenchmark(base.GenericBenchmark):
    def __init__(self, *args, fresh_inputs=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.fresh_inputs = fresh_inputs

    def get_latency(self, op, *args, **kwargs):
        if self.fresh_inputs:
            return self._get_fresh_input_latency(op, args, kwargs)
        return super().get_latency(op, *args, **kwargs)

    def _run_profile_cases(self, case_ids):
        if not self.fresh_inputs:
            return super()._run_profile_cases(case_ids)
        if self.is_backward:
            raise ValueError("Fresh-input backward profiling is not supported")
        cases = self._collect_cases()
        base.Config.available_case_ids.update(case.case_id for case in cases)
        selected = set(case_ids)
        executed = []
        for case in cases:
            if case.case_id not in selected:
                continue
            args, kwargs = self.unpack_to_args_kwargs(self.build_inputs(case))
            op, dispatch, _ = self._candidate_call()
            with dispatch:
                for _ in range(base.Config.profile_warmup):
                    fresh_args, fresh_kwargs = _clone_benchmark_inputs((args, kwargs))
                    op(*fresh_args, **fresh_kwargs)
                base.torch_device_fn.synchronize()
                for _ in range(base.Config.profile_iterations):
                    # Stateful inputs must be restored outside every capture.
                    fresh_args, fresh_kwargs = _clone_benchmark_inputs((args, kwargs))
                    base.torch_device_fn.synchronize()
                    hook = base.Config.profile_hook
                    scope = hook(backend=base.vendor_name, case_id=case.case_id) if hook else None
                    with scope if scope is not None else nullcontext():
                        op(*fresh_args, **fresh_kwargs)
                        base.torch_device_fn.synchronize()
            executed.append(case.case_id)
        base.Config.executed_case_ids.update(executed)
        return executed

    def set_shapes(self, shape_file_path=None, *, default_shapes=None):
        if default_shapes is None:
            return super().set_shapes(shape_file_path)
        with open(shape_file_path) as stream:
            configured = yaml.safe_load(stream) or {}
        selected = configured.get(self.op_name, configured.get(type(self).__name__, {}))

        def as_tuple(value):
            return tuple(as_tuple(item) for item in value) if isinstance(value, (tuple, list)) else value

        self.shapes = [as_tuple(shape) for shape in selected.get("shapes", default_shapes)]
        self.shape_desc = selected.get("shape_desc", self.DEFAULT_SHAPE_DESC)

    def _get_fresh_input_latency(self, op, args, kwargs):
        # Each sample starts from an independent snapshot. Input restoration
        # happens before the start event/clock, never inside the measured call.
        if self.is_backward:
            raise ValueError("Fresh-input backward measurement is not supported")
        if base.Config.mode == consts.BenchMode.CUDAGRAPH:
            raise ValueError(
                "State-changing benchmarks require fresh inputs; use kernel or operator mode"
            )
        if base.Config.mode not in (
            consts.BenchMode.KERNEL,
            consts.BenchMode.OPERATOR,
            consts.BenchMode.WRAPPER,
        ):
            raise ValueError("Undefined Value of Benchmark Mode.")
        device_timing = base.Config.mode == consts.BenchMode.KERNEL
        if device_timing and not hasattr(base.torch_device_fn, "Event"):
            raise ValueError("Backend has no event timer for fresh-input measurements")

        def sample():
            fresh_args, fresh_kwargs = _clone_benchmark_inputs((args, kwargs))
            base.torch_device_fn.synchronize()
            if device_timing:
                start = base.torch_device_fn.Event(enable_timing=True)
                end = base.torch_device_fn.Event(enable_timing=True)
                start.record()
                result = op(*fresh_args, **fresh_kwargs)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                start = time.perf_counter()
                result = op(*fresh_args, **fresh_kwargs)
                if base.Config.mode == consts.BenchMode.OPERATOR:
                    base.torch_device_fn.synchronize()
                elapsed = (time.perf_counter() - start) * 1000
            del result
            if elapsed <= 0:
                raise RuntimeError("Fresh-input timer returned a nonpositive latency")
            return elapsed

        warmup_elapsed = 0.0
        while warmup_elapsed < base.Config.warm_up:
            warmup_elapsed += sample()
        measured = []
        elapsed = 0.0
        while not measured or elapsed < base.Config.repetition:
            value = sample()
            measured.append(value)
            elapsed += value
        return sum(measured) / len(measured)
