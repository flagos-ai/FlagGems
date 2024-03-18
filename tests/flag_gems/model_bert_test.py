import torch
import flag_gems
from transformers import AutoTokenizer, BertConfig, BertModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
config = BertConfig()
model = BertModel(config)
model.to("cuda")
model.eval()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
with torch.no_grad():
    ref_outputs = model(**inputs)

print("=======================================")
flag_gems.enable()
with torch.no_grad():
    res_outputs = model(**inputs)

maxdiff = torch.max(
    torch.abs(ref_outputs.last_hidden_state - res_outputs.last_hidden_state)
)

assert torch.allclose(
    ref_outputs.last_hidden_state, res_outputs.last_hidden_state, atol=1e-2, rtol=1e-2
), f"REF: {ref_outputs.last_hidden_state}\nRES: {res_outputs.last_hidden_state}\nMAXDIFF: {maxdiff}"

print("##### SUCCEED #####")
