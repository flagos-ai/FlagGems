import torch
from vllm import LLM, SamplingParams
from vllm.attention.selector import global_force_attn_backend
from vllm.platforms import _Backend

import flag_gems

global_force_attn_backend(_Backend.TRITON_MLA)


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=120)
    model_name = "deepseek-ai/DeepSeek-V3"
    llm = LLM(
        model=model_name,
        max_model_len=2048,
        gpu_memory_utilization=0.98,
        enforce_eager=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    flag_gems.enable()
    with torch.no_grad():
        outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
