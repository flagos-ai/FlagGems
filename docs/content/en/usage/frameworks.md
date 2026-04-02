---
title: Integration with Frameworks
weight: 60
---

# Integration with Popular Frameworks

To help integrate *FlagGems* into real-world scenarios, we provide examples
with widely-used deep learning frameworks.
These integrations require minimal code changes and preserve the original workflow structure.

For full examples, see the [`examples/`](https://github.com/flagos-ai/FlagGems/tree/master/examples)
directory in the source code repository.

## 1. Hugging Face Transformers

Integration with Hugging Face's [`transformers` library](https://github.com/huggingface/transformers)
is straightforward.

During inference, you can activate the *FlagGems* acceleration
without modifying the model or the tokenizer logic.
Here's a minimal example:

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

This pattern ensures that all compatible operators used during generation
are  automatically accelerated. You can find more examples in the following
files:

- `examples/model_llama_test.py`
- `examples/model_llava_test.py`

## 2. vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput inference engine
designed for serving large language models efficiently.
It supports features like paged attention, continuous batching, and optimized memory management.

*FlagGems* can be integrated into vLLM to replace both standard PyTorch (`aten`) operators
and vLLM's internal custom kernels.

### 2.1 Replacing standard PyTorch operators in vLLM

To accelerate the standard PyTorch operators (e.g., `add`, `masked_fill`) in vLLM,
you can use the same approach as you do in other frameworks.
By invoking `flag_gems.enable()` before model initialization or inference,
you can override all compatible PyTorch `aten` operators,
including operators that are indirectly used in vLLM.

### 2.2 Replacing vLLM-Specific Custom Operators

To further optimize the internal kernels in vLLM, *FlagGems* provides an additional API:

```python
flag_gems.apply_gems_patches_to_vllm(verbose=True)
```

This API patches certain vLLM-specific C++ or Triton operators with the *FlagGems* implementations.
When `verbose` is set to `True`, the invocation records which functions
have been replaced:

```none
Patched RMSNorm.forward_cuda with FLAGGEMS custom_gems_rms_forward_cuda
Patched RotaryEmbedding.forward_cuda with FLAGGEMS custom_gems_rope_forward_cuda
Patched SiluAndMul.forward_cuda with FLAGGEMS custom_gems_silu_and_mul
```

Use this when more comprehensive `flag_gems` coverage is desired.

### 2.3 Full example

```python
from vllm import LLM, SamplingParams
import flag_gems

# Step 1: enable acceleration for PyTorch (aten) operators
flag_gems.enable()

# Step 2: (optional) patch vLLM custom ops
flag_gems.apply_gems_patches_to_vllm(verbose=True)

# Step 3: use vLLM as usual
llm = LLM(model="sharpbai/Llama-2-7b-hf")
sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

output = llm.generate("Tell me a joke.", sampling_params)
print(output)
```

## 3. Megatron-LM

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) is a highly optimized framework
for large-scale language model pre-training and fine-tuning.
Due to its tight integration with custom training loops and internal utilities,
integrating *FlagGems* into Megatron-LM requires a slightly more targeted approach.

Since the training loop in Megatron is tightly bound to distributed data loading,
gradient accumulation, and pipeline parallelism, we recommend applying *FlagGems*
accelerations only for the forward and backward computation stages.

### 3.1 Recommended Integration Point

The most reliable way to use FlagGems in Megatron-LM is by modifying the `train_step` function
in [`megatron/training.py`](https://github.com/NVIDIA/Megatron-LM/blob/v2.6/megatron/training.py#L347).
Specifically, wrap the block where `forward_backward_func` is invoked as shown below:

```python
def train_step(forward_step_func, data_iterator,
               model, optimizer, lr_scheduler):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # Set grad to zero.
    if args.DDP_impl == 'local' and args.use_contiguous_buffers_in_local_ddp:
        for partition in model:
            partition.zero_grad_buffer()
    optimizer.zero_grad()

    # Forward pass with flag_gems acceleration
    import flag_gems
    with flag_gems.use_gems():
      forward_backward_func = get_forward_backward_func()
      losses_reduced = forward_backward_func(
          forward_step_func, data_iterator, model,
          optimizer, timers, forward_only=False)
      )

    # Empty unused memory
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()
    ...
```

This ensures that only the forward and backward computation logic are executed
with *FlagGems* acceleration, while other components such as data loading
and optimizer steps remain unchanged.

### 3.2 Scope and Limitations

> [!WARNING]
> **Warning**
>
> The Megatron-LM project constantly evolves over time. Please use caution
> when identifying the integration point.

While `flag_gems.enable()` is sufficient in most frameworks, we observed that
applying it early in Megatron-LM's pipeline can sometimes cause unexpected behavior,
especially during the data loading phase.
For better stability, we recommend using `flag_gems.use_gems()` as a context manager
to be applied to the computation stage.

If you wish to accelerate a broader range of components (e.g., optimizer, preprocessing),
you may try enabling `flag_gems` globally with `flag_gems.enable()`.
However, this approach is not thoroughly tested and may require additional validation
based on your Megatron-LM version.

We encourage community [contributions](/FlagGems/contribution/overview/) —
please consider open an issue or submit a PR to help us achieve a broader
Megatron-LM integration.
