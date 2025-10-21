from pymodel import DecoderGPT
import torch


TOKENIZER_PATH = "model/custom_tokenizer_mental.json"

from transformers import PreTrainedTokenizerFast
tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER_PATH)
vocab_size = tok.vocab_size

d_model = context_length= 256
n_heads = 2
n_layers = 4

device = "cuda" if torch.cuda.is_available() else "cpu"


class Slm:
    def __init__(self):
        self.m = DecoderGPT(vocab_size=vocab_size, d_model=d_model, n_heads=n_heads, n_layers=n_layers, context_length=context_length, dropout=0.05).to(device)
        checkpoint = torch.load("model/best_model.pt", map_location="cuda" if torch.cuda.is_available() else "cpu")
        self.m.load_state_dict(checkpoint['model_state_dict'])


    def generate(self, input_text, max_new_tokens=40):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.m.to(device)
        self.m.eval()
        with torch.no_grad():
            input_ids = torch.tensor(tok.encode(input_text), dtype=torch.long, device=device).unsqueeze(0)
            output = self.m.generate(input_ids, max_new_tokens=max_new_tokens)
            generated_text = tok.decode(output[0].tolist())
        return generated_text