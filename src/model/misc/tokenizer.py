import json

class CharTokenizer:
    def __init__(self, transcripts=None, vocab_dict=None):
        self.blank_id = 0
        
        if vocab_dict:
            self.char2id = vocab_dict
            self.id2char = {v: k for k, v in self.char2id.items()}
            self.chars = sorted(list(self.char2id.keys()))
            
        elif transcripts:
            chars = set("".join(transcripts))
            self.chars = sorted(list(chars))
            self.char2id = {c: i+1 for i, c in enumerate(self.chars)}
            self.id2char = {i+1: c for i, c in enumerate(self.chars)}
            
        else:
            raise ValueError("Harus menyediakan 'transcripts' (untuk train) atau 'vocab_dict' (untuk load).")
        
    def text_to_int(self, text):
        return [self.char2id[c] for c in text]

    def int_to_text(self, ids):
        return "".join([self.id2char[i] for i in ids if i != 0])
        
    def vocab_size(self):
        return len(self.char2id) + 1
    
    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.char2id, f, ensure_ascii=False, indent=4)
            
    @classmethod
    def load(cls, path):
        with open(path, 'r', encoding='utf-8') as f:
            vocab_dict = json.load(f)
        return cls(vocab_dict=vocab_dict)