import torch
from pytorch_transformers import *
from pathlib import Path

sample_text = "Рад познакомиться с вами."
my_model_dir = str(Path.home() / "pytorch-rubert")

tokenizer = BertTokenizer.from_pretrained(my_model_dir)
model = BertModel.from_pretrained(my_model_dir, output_hidden_states=True)

input_ids = torch.tensor([tokenizer.encode(sample_text, add_special_tokens=True)])
print(f"Input ids: {input_ids}")
with torch.no_grad():
    last_hidden_states = model(input_ids)[0]
    print(f"Shape of last hidden states: {last_hidden_states.shape}")
    print(last_hidden_states)