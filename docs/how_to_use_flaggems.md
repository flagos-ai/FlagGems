# How To Use FlagGems

## Basic Usage

To use the *FlagGems* operator library, import `flag_gems` and enable acceleration
before running your program. You can enable it globally, selectively, or temporarily.

### Option 1: Global Enablement

To apply *FlagGems* optimizations across your entire script or your interactive session:

```python
import flag_gems

# Enable all FlagGems operators globally
flag_gems.enable()
```

Once enabled, all supported operators in your code will be replaced automatically
by the optimized *FlagGems* implementations — no further changes needed.

### Option 2: Selective Enablement

To enable only specific operators and skip the rest:

```python
import flag_gems

# Enable only selected ops
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

This is useful when you want to accelerate only a subset of operators.

### Option 3: Scoped Enablement

For finer controls, you can enable *FlagGems* only within a specific code block
using a context manager:

```python
import flag_gems

# Enable flag_gems temporarily
with flag_gems.use_gems():
    # Code inside this block will use FlagGems-accelerated operators
    ...
```

This scoped usage is useful when you want to:

- perform performance benchmarks, or
- compare correctness between implementations, or
- apply acceleration selectively in complex workflows.

You can also use selective enablement in a context manager:

```python
# Enable only specific ops in the scope
with flag_gems.use_gems(include=["sum", "add"]):
    # Only sum and add will be accelerated
    ...

# Or exclude specific ops
with flag_gems.use_gems(exclude=["mul", "div"]):
    # All operators except mul and div will be accelerated
    ...
```

Note: The `include` parameter has higher priority than `exclude`.
If both are provided, `exclude` is ignored.

## Advanced Usage

The `flag_gems.enable(...)` and `flag_gems.only_enable(...)` functions support several optional parameters
which give you finer-grained control over how acceleration is applied.
This allows for more flexible integration and easier debugging or profiling when working with complex workflows.

### Parameters

| Parameter      | Type      | Description                                         |
| -------------- | --------- | --------------------------------------------------- |
| `unused`       | List[str] | Disable specific operators (for `enable`)           |
| `include`      | List[str] | Enable only specific operators (for `only_enable`)  |
| `record`       | bool      | Log operator calls for debugging or profiling       |
| `path`         | str       | Log file path (only used when `record=True`)        |

### Example : Selectively Disable Specific Operators

You can use the `unused` parameter in `enable()` to exclude certain operators from being accelerated by `FlagGems`.
This is especially useful when a particular operator does not behave as expected in your workload,
or if you're seeing suboptimal performance and want to use the original implementation.

```python
flag_gems.enable(unused=["sum", "add"])
```

With this configuration, `sum` and `add` will continue to use the native PyTorch implementations,
while all other supported operators will use the *FlagGems* version.

### Example : Selectively Enable Specific Operators

You can use `only_enable()` with the `include` parameter to accelerate only a subset of operators:

```python
flag_gems.only_enable(include=["rms_norm", "softmax"])
```

This registers only the specified operators, skipping all the others.

### Example : Enable Debug Logging

If you want to log the operator usage during runtime, you can
set `record=True` along with `path` set to  the path of the log file.

```python
flag_gems.enable(
    record=True,
    path="./gems_debug.log"
)
```

After running your script, inspect the log file (e.g., `./gems_debug.log`) to check
the list of operators that have been invoked through `flag_gems`.

Sample log content:

```shell
$ cat ./gems_debug.log
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.fill: GEMS FILL_SCALAR_
[DEBUG] flag_gems.ops.mm: GEMS MM
[DEBUG] flag_gems.fused.reshape_and_cache: GEMS RESHAPE_AND_CACHE
```

### Example : Query Registered Operators

After enabling *FlagGems*, you can check the operators registered:

```python
import flag_gems

flag_gems.enable()

# Get list of registered function names
registered_funcs = flag_gems.all_registered_ops()
print("Registered functions:", registered_funcs)

# Get list of registered operator keys
registered_keys = flag_gems.all_registered_keys()
print("Registered keys:", registered_keys)
```

This is useful for debugging or verifying which operators are active.

## Running FlagGems on Non-NVIDIA Hardware

### Supported Platforms

FlagGems supports a range of AI chips beyond NVIDIA.
For an up-to-date list of validated platforms, please refer to
[Supported Platforms](./features.md#platforms-supported)

### Unified Usage Interface

Regardless of the underlying hardware, the usage of `flag_gems` remains exactly the same.
There is no need to modify application code when switching from NVIDIA to non-NVIDIA platforms.

Once you call `import flag_gems` and enable acceleration via `flag_gems.enable()`,
operator dispatch will be automatically routed to the correct backend.
This provides a consistent developer experience across different environments.

### Backend Requirements

Although the usage pattern is unchanged, there are some prerequisites when running *FlagGems* on non-NVIDIA platforms.
The **PyTorch** and the **Triton compiler** have to be installed and properly configured on the target platform.

There are two common ways to obtain compatible builds:

1. **Consult your hardware vendor**

   Hardware vendors typically maintain custom builds of PyTorch and Triton tailored to their chips.
   Contact the vendor to request the appropriate versions.

2. **Explore the FlagTree project**

   The [FlagTree](https://github.com/flagos-ai/flagtree) project offers a unified Triton compiler
   that supports a range of AI chips, including NVIDIA and non-NVIDIA platforms.
   It consolidates vendor-specific patches and enhancements into a shared, open-source backend,
   simplifying compiler maintenance and enabling multi-platform compatibility.

   > [!Note]
   > FlagTree provides Triton only. A matching PyTorch build is still required separately.

> [!Note]
> Some platforms may require additional setup or patching.

### Backend Auto-Detection and Manual Setting

By default, *FlagGems* automatically detects the current hardware backend at runtime
and selects the corresponding implementation.
In most cases, no manual configuration is required, and everything works out of the box.

However, if auto-detection fails or there are compatibility issues in your environment,
you can manually set the target backend to ensure correct runtime behavior.
To do this, set the following environment variable before running your code:

```shell
export GEMS_VENDOR=<your_vendor_name>
```

> ⚠️  This setting should match the actual hardware platform.
> Manually setting an incorrect backend may result in runtime errors.

You can verify the active backend at runtime using:

```python
import flag_gems
print(flag_gems.vendor_name)
```

## Integration with Popular Frameworks

To help integrate *FlagGems* into real-world scenarios, we provide examples
with widely-used deep learning frameworks.
These integrations require minimal code changes and preserve the original workflow structure.

For full examples, see the [`examples/`](https://github.com/flagos-ai/FlagGems/tree/master/examples)
directory in the source code repository.

### Example 1: Hugging Face Transformers

Integration with Hugging Face's `transformers` library is straightforward.
You can follow the basic usage patterns introduced in previous sections.

During inference, you can activate acceleration without modifying the model
or tokenizer logic. Here's a minimal example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import flag_gems

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

# Move model to correct device and set to eval mode
device = flag_gems.device
model.to(device).eval()

# Prepare input and run inference with flag_gems enabled
inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
with flag_gems.use_gems():
    output = model.generate(**inputs, max_length=100, num_beams=5)
```

This pattern ensures that all compatible operators used during generation will be automatically accelerated.
You can find more examples in the following files:

- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

### Example 2: vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine
designed for serving large language models efficiently.
It supports features like paged attention, continuous batching, and optimized memory management.

*FlagGems* can be integrated into vLLM to replace both standard PyTorch (`aten`) operators
and vLLM's internal custom kernels.

#### Replacing Standard PyTorch Operators in vLLM

To accelerate standard PyTorch operators (e.g., `add`, `masked_fill`) in vLLM,
you can use the same approach as you do in other frameworks.
By invoking `flag_gems.enable()` before model initialization or inference.
you override all compatible PyTorch `aten` operators, including those indirectly used in vLLM.

#### Replacing vLLM-Specific Custom Operators

To further optimize vLLM’s internal kernels, *FlagGems* provides an additional API:

```python
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

This API patches certain vLLM-specific C++ or Triton operators with FlagGems implementations.
When `verbose` is set to `True`, the invocation records functions that were replaced:

```none
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```

Use this when more comprehensive `flag_gems` coverage is desired.

#### Full Example: Enable `flag_gems` in vLLM Inference

```python
from vllm import LLM, SamplingParams
import flag_gems

# Step 1: Enable acceleration for PyTorch (aten) operators
flag_gems.enable()

# Step 2: (Optional) Patch vLLM custom ops
flag_gems.apply_gems_patches_to_vllm(verbose=True)

# Step 3: Use vLLM as usual
llm = LLM(model="sharpbai/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

output = llm.generate("Tell me a joke.", sampling_params)
print(output)
```

### Example 3: Megatron

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is a highly optimized framework
for large-scale language model pretraining and fine-tuning.
Due to its tight integration with custom training loops and internal utilities,
integrating *FlagGems* into Megatron requires a slightly more targeted approach.

Since the training loop in Megatron is tightly bound to distributed data loading,
gradient accumulation, and pipeline parallelism, we recommend applying *FlagGems*
accelerations only for the forward and backward computation stages.

#### Recommended Integration Point

The most reliable way to use FlagGems in Megatron is by modifying the `train_step` function
in [`megatron/training/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/training/training.py#L1360).
Specifically, wrap the block where `forward_backward_func` is invoked as shown below:

```python
def train_step(forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config):
    """Single training step."""
    args = get_args()
    timers = get_timers()

     # CUDA Graph capturing logic omitted
    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
    	# Gradient zeroing logic omitted

        # Forward pass with flag_gems acceleration
        import flag_gems
        with flag_gems.use_gems():
          forward_backward_func = get_forward_backward_func()
          losses_reduced = forward_backward_func(
              forward_step_func=forward_step_func,
              data_iterator=data_iterator,
              model=model,
              num_microbatches=get_num_microbatches(),
              seq_length=args.seq_length,
              micro_batch_size=args.micro_batch_size,
              decoder_seq_length=args.decoder_seq_length,
              forward_only=False,
              adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
          )

    should_checkpoint, should_exit, exit_code = rerun_state_machine.should_checkpoint_and_exit()
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Other post-step operations omitted
```

This ensures that only the forward and backward computation logic runs with *FlagGems* acceleration,
while other components such as data loading and optimizer steps remain unchanged.

#### Scope and Limitations

While `flag_gems.enable()` is sufficient in most frameworks, we observed that
applying it early in Megatron’s pipeline can sometimes cause unexpected behavior,
especially during the data loading phase.
For better stability, we recommend using `flag_gems.use_gems()` as a context manager
to be applied to the computation stage.

If you wish to accelerate a broader range of components (e.g., optimizer, preprocessing),
you may try enabling `flag_gems` globally with `flag_gems.enable()`.
However, this approach is not thoroughly tested and may require additional validation
based on your Megatron version.

We encourage community contributions — please open an issue or submit a PR
to help improve broader Megatron integration.

## Multi-GPU Deployment

In real-world LLM deployment scenarios, multi-GPU or multi-node setups are often required
to support large model sizes and high-throughput inference.
*FlagGems* supports these scenarios by accelerating operator execution across multiple GPUs.

### Single-Node vs Multi-Node Usage

For **single-node deployments**, the integration is straightforward. You can import `flag_gems`
and invoke `flag_gems.enable()` at the beginning of your script.
This enables acceleration without requiring any additional changes.

In **multi-node deployments**, however, this approach is insufficient.
Distributed inference frameworks (like vLLM) spawn multiple worker processes across nodes,
where every process must initialize `flag_gems` individually.
If the activation occurs only in the launch script on one node, worker processes
on other nodes will fall back to the default implementation which is not accelerated.

### Integration Example: vLLM + DeepSeek

To enable *FlagGems* in a distributed vLLM + DeepSeek deployment:

1. **Baseline Verification**

   Before integrating *FlagGems*, verify that the model can load and serve correctly without it.
   For example, loading a model like `Deepseek-R1` typically requires **at least two H100 GPUs**
   and it can take **up to 20 minutes** to initialize, depending on the checkpoint size and
   the system I/O performance.

1. **Inject `flag_gems` into vLLM Worker Code**

   Locate the appropriate model runner script depending on your vLLM version:

   - If you are using the **vLLM v1 architecture** (available in vLLM ≥ 0.8),
     modify `vllm/v1/worker/gpu_model_runner.py`
   - If you are using the **legacy v0 architecture**, modify `vllm/worker/model_runner.py`

   In either file, insert the following logic after the last `import` statement:

   ```python
   import os
   if os.getenv("USE_FLAGGEMS", "false").lower() in ("1", "true", "yes"):
       try:
           import flag_gems
           flag_gems.enable()
           flag_gems.apply_gems_patches_to_vllm(verbose=True)
           logger.info("Successfully enabled flag_gems as default ops implementation.")
       except ImportError:
           logger.warning("Failed to import 'flag_gems'. Falling back to default implementation.")
       except Exception as e:
           logger.warning(f"Failed to enable 'flag_gems': {e}. Falling back to default implementation.")
   ```

1. **Set Environment Variables on All Nodes**

   Before launching the service, ensure all nodes have the following environment variable set:

   ```shell
   export USE_FLAGGEMS=1
   ```

1. **Start Distributed Inference and Verify**

   Launch the service and check the startup logs on each node for messages
   indicating that operators have been overridden.

   ```none
   Overriding a previously registered kernel for the same operator and the same dispatch key
   operator: aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
     registered at /pytorch/build/aten/src/ATen/RegisterSchema.cpp:6
   dispatch key: CUDA
   previous kernel: registered at /pytorch/aten/src/ATen/LegacyBatchingRegistrations.cpp:1079
        new kernel: registered at /dev/null:488 (Triggered internally at /pytorch/aten/src/ATen/core/dispatch/OperatorEntry.cpp:154.)
   self.m.impl(
   ```

   This confirms that `flag_gems` has been successfully enabled across all nodes.

## Building Custom Models Using Gems Operators

In some scenarios, users may wish to build their own models from scratch
or to adapt existing ones to better suit their specific use cases.
To support this, *FlagGems* provides a growing collection of high-performance modules
commonly used in large language models (LLMs).

These components are implemented using *FlagGems*-accelerated operators
and can be used in the way you use any standard `torch.nn.Module`.
You can seamlessly integrate them into your architecture to benefit from kernel-level acceleration
without writing custom CUDA or Triton code.

Modules can be found in
[flag_gems/modules](https://github.com/flagos-ai/FlagGems/tree/master/src/flag_gems/modules).

### Modules Available

| Module                 | Description                                           | Supported Features                         |
| ---------------------- | ----------------------------------------------------- | ------------------------------------------ |
| `GemsRMSNorm`          | RMS LayerNorm                                         | Fused residual add, `inplace` & `outplace` |
| `GemsRope`             | Standard rotary position embedding                    | `inplace` & `outplace`                     |
| `GemsDeepseekYarnRoPE` | RoPE with extrapolation for DeepSeek-style LLMs       | `inplace` & `outplace`                     |
| `GemsSiluAndMul`       | Fused SiLU activation with elementwise multiplication | `outplace` only                            |

We encourage users to use these as drop-in replacements for equivalent PyTorch layers.
More components such as fused attention, MoE layers, and transformer blocks are under development.

## Achieving Optimal Performance with Gems

While *FlagGems* kernels are designed for high performance, achieving optimal end-to-end speed
in full model deployments requires careful integration and consideration of runtime behavior.
In particular, two common performance bottlenecks are:

- **Runtime autotuning overhead** in production environments.
- **Suboptimal dispatching** due to framework-level kernel registration or interaction with the Triton runtime.

These issues can occasionally offset the benefits of highly optimized kernels.
To address them, we provide two complementary optimization paths designed to ensure that
*FlagGems* operates at peak efficiency in real inference scenarios.

### Pre-tuning Model Shapes for Inference Scenarios

*FlagGems* provides [`LibTuner`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/utils/libentry.py#L139),
a lightweight enhancement to Triton’s autotuning system.
`LibTuner` introduces a **persistent, per-device tuning cache** that
helps mitigate runtime overhead in the Triton’s default autotuning process.

#### Why Pre-tuning?

Triton typically performs autotuning during the first few executions of a new input shape,
which may cause latency spikes—especially in latency-sensitive inference systems.
`LibTuner` addresses this with:

- _Persistent caching_: Best autotune configs are saved across runs.
- _Cross-process sharing_: Cache is shared across processes on the same device.
- _Reduced runtime overhead_: Once tuned, operators skip tuning in future runs.

This is particularly useful for operators like `mm` and `addmm`,
which often trigger the autotune logic in Triton.

#### How to Use Pre-tuning

To proactively warm up your system and to populate the cache:

1. Identify key input shapes used in your production workload.
1. Run the pre-tuning script to benchmark and cache best configs.
   You can run `python examples/pretune.py` as an example.
1. Deploy normally, and *FlagGems* will automatically pick the optimal config
   from cache during inference.

> ![Tip]
> - The `pretune.py` script accepts example shapes and workloads
>   which can be used to simulate your model's actual use cases.
>   You can customize it for batch sizes, sequence lengths, etc.
> - In frameworks like **vLLM** (`v0.8.5+`), enabling `--compile-mode`
>   automatically performs a warmup step.
>   If *FlagGems*  is enabled, this also triggers `LibTuner`-based
>   pre-tuning implicitly.

For more details (e.g. customizing your tuning cache path and settings),
refer to the [`examples/pretune.py`](https://github.com/flagos-ai/FlagGems/blob/master/examples/pretune.py)
as an example.

### Using C++-Based Operator Wrappers for Further Performance Gains

Another advanced optimization path with *FlagGems* is the use of **C++ wrappers** for selected operators.
While Triton kernels offer reasonably good compute performance, Triton itself is a Python-embedded DSL.
This means that both the operator definitions and the runtime dispatches are in Python,
which can introduce **non-trivial overhead** in latency-sensitive or high-throughput scenarios.

To address this, *FlagGems* provides a C++ runtime solution that encapsulates the operator’s wrapper logic,
registration mechanism, and runtime management entirely in C++,
while still reusing the underlying Triton kernels for the actual computation.
This approach preseves Triton's kernel-level efficiency while significantly reducing Python-related overhead,
enabling tighter integration with low-level CUDA workflows and improving overall inference performance.

#### Installation & Setup

To use the C++ operator wrappers:

1. Follow the [installation guide](./installation.md) to compile
   and install the C++ version of `flag_gems`.

1. Verify that the installation is successful using the following snippet:

   ```python
   try:
       from flag_gems import c_operators
       has_c_extension = True
   except Exception as e:
       c_operators = None  # avoid import error if c_operators is not available
       has_c_extension = False
   ```

   If `has_c_extension` is `True`, then the C++ runtime path is available.

1. When installed successfully, C++ wrappers will be preferred **in patch mode**
   and when explicitly building models using *FlagGems*-defined modules.

   For example, `gems_rms_forward` will by default use the C++ wrapper version of `rms_norm`.
   You can refer to the actual usage in the
   [`normalization.py`](https://github.com/flagos-ai/FlagGems/blob/master/src/flag_gems/modules/normalization.py#L46)
   to better understand how C++ operator wrappers are integrated and invoked.

#### Explicitly Using C++ Operators

If you want to *invoke the C++-wrapped operators directly*, bypassing any patch logics
or fall back paths, you can use the `torch.ops.flag_gems` namespace as shown below:

```python
output = torch.ops.flag_gems.fused_add_rms_norm(...)
```

This gives you *precise control* over operator dispatch, which can be beneficial
in performance-sensitive contexts.

#### Currently Supported C++-Wrapped Operators

| Operator Name        | Description                              |
| -------------------- | ---------------------------------------- |
| `add`                | Element-wise addition                    |
| `bmm`                | Batch Matrix Multiplication              |
| `cat`                | Concatenate                              |
| `fused_add_rms_norm` | Fused addition + RMSNorm                 |
| `mm`                 | Matrix multiplication                    |
| `nonzero`            | Returns the indices of non-zero elements |
| `rms_norm`           | Root Mean Square normalization           |
| `rotary_embedding`   | Rotary position embedding                |
| `sum`                | Reduction across dimensions              |

We are actively expanding this list as part of our ongoing performance roadmap.
