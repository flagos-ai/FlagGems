# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import flag_gems

device = flag_gems.device


@pytest.mark.parametrize(
    "prompt",
    ["How are you today?", "What is your name?", "Who are you?", "Where are you from?"],
)
def test_accuracy_llama(prompt):
    tokenizer = AutoTokenizer.from_pretrained("sharpbai/Llama-2-7b-hf")
    model = AutoModelForCausalLM.from_pretrained("sharpbai/Llama-2-7b-hf")

    model.to(device).eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device=device)
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
