from transformers import GPT2Model, GPT2Config
import torch

# Define the configuration for the GPT-2 model with multi-headed self-attention
config = GPT2Config(
    n_heads=8,  # Number of attention heads
    n_embd=768  # Dimension of the model's hidden states
)

# Initialize the GPT-2 model with the specified configuration
model = GPT2Model(config)

# Example input tensor
input_ids = torch.tensor([[1, 2, 3, 4, 5]])

# Forward pass through the model to obtain the output with multi-headed self-attention
outputs = model(input_ids)
print(outputs)