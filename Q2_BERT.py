from transformers import AutoModel, AutoTokenizer
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"  # Example: BERT model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Encode text
input_text = "Example input text"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Get the encoder output
with torch.no_grad():
    output = model(input_ids)

# The output contains the hidden states, pooled output, etc.
# For example, to get the hidden states of all layers
hidden_states = output.last_hidden_state
print(hidden_states)

#####################Printed Output#######################################
#tensor([[[-0.2796,  0.0169, -0.2637,  ..., -0.7017,  0.0977,  0.8031],
#         [-0.1798,  0.1107, -0.7470,  ..., -0.3447,  0.5448,  0.1668],
#        [-1.2174,  0.5254, -0.5268,  ..., -1.1756,  0.0154,  0.6167],
#         [-0.1411, -0.0105,  0.0791,  ..., -1.0269, -0.7315,  0.5230],
#        [ 0.9685,  0.0289, -0.5429,  ...,  0.2606, -0.7593, -0.2211]]])
#####################Printed Output#######################################