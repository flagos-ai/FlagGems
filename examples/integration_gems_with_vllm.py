# SPDX-License-Identifier: Apache-2.0
import random
import time

import numpy as np
import torch
from vllm import LLM, SamplingParams

import flag_gems

flag_gems.enable()  # Enable gems for PyTorch (aten) operators
flag_gems.apply_gems_patches_to_vllm(verbose=True)  # Patch vLLM custom ops


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=120)


def main():
    # Create an LLM.
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        tensor_parallel_size=1,
        max_model_len=1024,
        gpu_memory_utilization=0.98,
        # enforce_eager=True,
    )

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    time.sleep(10)


if __name__ == "__main__":
    main()
