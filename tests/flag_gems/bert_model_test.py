import torch
import flag_gems
from transformers import AutoTokenizer, BertForPreTraining, BertConfig, BertModel

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
# model = BertForPreTraining.from_pretrained("google-bert/bert-base-uncased")
config = BertConfig()
model = BertModel(config)
model.to("cuda")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to("cuda")
outputs = model(**inputs)
