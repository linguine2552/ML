from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datetime import datetime

tokenizer = AutoTokenizer.from_pretrained("MaziyarPanahi/Calme-7B-Instruct-v0.2")
model = AutoModelForCausalLM.from_pretrained("MaziyarPanahi/Calme-7B-Instruct-v0.2")

prompt = "generate 6 random words and pick 2"
input_ids = tokenizer.encode(prompt, return_tensors="pt")

output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    top_k=0,  # Disable top_k sampling - model will consider all possible next tokens, regardless of their probability
    top_p=1.0,  # Set top_p to 1.0 for maximum diversity - model will consider all possible next tokens that contribute to the cumulative probability mass of 1.0
    do_sample=True,  # Enable sampling - enable sampling instead of greedy decoding - allows the model to generate more diverse and random words
    temperature=1.0,  # Set temperature to 1.0 for maximum randomness
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

current_date = datetime.now().strftime("%m-%d-%Y")

with open(f"{current_date}.merc", "w") as file:
    file.write(generated_text)