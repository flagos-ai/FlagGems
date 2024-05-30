import torch
import pytest
import copy
import flag_gems
from transformers import AutoTokenizer, AutoModelForCausalLM


@pytest.mark.parametrize(
    "prompt",
    ["How are you today?", "What is your name?", "Who are you?", "Where are you from?"],
)
def test_accuracy_llama(prompt):
    tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

    model.to("cuda").eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device="cuda")
    with torch.no_grad():
        ref_output = model.generate(**inputs, max_length=100, num_beams=5)

    with flag_gems.use_gems():
        res_output = model.generate(**inputs, max_length=100, num_beams=5)

    maxdiff = torch.max(torch.abs(ref_output - res_output))
    assert torch.allclose(
        ref_output,
        res_output,
        atol=1e-3,
        rtol=1e-3,
    ), f"LLAMA FAIL with maxdiff {maxdiff} \nREF: {ref_output}\nRES: {res_output}"
