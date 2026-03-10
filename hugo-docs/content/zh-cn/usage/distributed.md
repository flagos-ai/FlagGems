---
title: 分布式环境
weight: 70
---

# Multi-GPU Deployment

In real-world LLM deployment scenarios, multi-GPU or multi-node setups are often required
to support large model sizes and high-throughput inference.
*FlagGems* supports these scenarios by accelerating operator execution across multiple GPUs.

## Single-Node vs Multi-Node Usage

For **single-node deployments**, the integration is straightforward. You can import `flag_gems`
and invoke `flag_gems.enable()` at the beginning of your script.
This enables acceleration without requiring any additional changes.

In **multi-node deployments**, however, this approach is insufficient.
Distributed inference frameworks (like vLLM) spawn multiple worker processes across nodes,
where every process must initialize `flag_gems` individually.
If the activation occurs only in the launch script on one node, worker processes
on other nodes will fall back to the default implementation which is not accelerated.

## Integration Example: vLLM + DeepSeek

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
