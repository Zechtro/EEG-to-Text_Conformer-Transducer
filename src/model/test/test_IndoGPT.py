from indobenchmark import IndoNLGTokenizer
from transformers import GPT2LMHeadModel
import torch

def explore():
    model_name = "indobenchmark/indogpt"
    
    # 1. Use the specific Indonesian Tokenizer
    print("Loading IndoNLGTokenizer...")
    tokenizer = IndoNLGTokenizer.from_pretrained(model_name)
    
    # 2. Load the model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()

    # 3. Test Tokenization
    phrase = "saya sedang belajar penelitian BCI"
    # IndoNLGTokenizer might require different call signatures, check help(tokenizer)
    ids = tokenizer.encode(phrase)
    
    print("\n--- TOKENIZATION TEST ---")
    print(f"Original: {phrase}")
    print(f"Tokens:   {tokenizer.convert_ids_to_tokens(ids)}")
    print(f"IDs:      {ids}")
    print(f"U-Dim:    {len(ids) + 1}")

    # 4. Fix Generation Test
    prompt = "Ibu kota Indonesia adalah"
    input_ids = torch.tensor([tokenizer.encode(prompt)]) # Wrap in list for batch dim
    
    if input_ids.shape[1] == 0:
        print("Error: Input IDs are still empty. Tokenizer configuration issue.")
        return

    print("\n--- GENERATION TEST ---")
    output = model.generate(
        input_ids, 
        max_length=15,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else 50256
    )
    print(f"Generated: {tokenizer.decode(output[0])}")

if __name__ == "__main__":
    explore()