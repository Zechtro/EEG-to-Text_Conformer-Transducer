from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("indobenchmark/indogpt")

# Step-by-step tokenization to see where it breaks
text = "saya belajar"
raw_tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(raw_tokens)

print(f"Text: {text}")
print(f"Tokens: {raw_tokens}")
print(f"IDs: {ids}")

# If that STILL returns [], try the "GPT2-Base" fallback with IndoGPT weights
if not ids:
    print("IndoGPT config failed. Falling back to standard GPT2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    ids = tokenizer.encode(text, add_special_tokens=False)
    print(f"GPT2 Fallback IDs: {ids}")