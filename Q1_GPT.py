
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Encode input text
input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate output
output = model.generate(input_ids, max_length=100, num_return_sequences=3, no_repeat_ngram_size=2, top_k=50)

# Decode and print the output
for i, sample_output in enumerate(output):
    print(f"Output {i+1}: {tokenizer.decode(sample_output, skip_special_tokens=True)}\n")
    
###
#Output 1: Once upon a time, the world was a place of great beauty and great danger.
#The world of the gods was the place where the great gods were born, and where they were to live.
#The world that was created was not the same as the one that is now.
#It was an endless, endless world. And the Gods were not born of nothing.
#They were created of a single, single thing. That was why the universe was so beautiful. Because the cosmos was made of two
