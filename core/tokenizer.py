import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from core.config import Config

class Tokenizer:
    def __init__(self, tokenizer_path=Config.tokenizer_path):

        self.tokenizer = None
        if os.path.exists(tokenizer_path):
            self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    
    def train(self, file:str=Config.tokenizer_data_path):
        
        tokenizer = BertWordPieceTokenizer(lowercase=True)

        tokenizer.train(
            files=file,
            vocab_size=Config.vocab_size,
            min_frequency=Config.min_frequency,
            special_tokens=list(Config.special_tokens)
        )

        if not os.path.exists(Config.tokenizer_path):
            os.makedirs(Config.tokenizer_path, exist_ok=True)

        tokenizer.save_model(Config.tokenizer_path)

        self.tokenizer = BertTokenizerFast.from_pretrained(Config.tokenizer_path)

        print(f"{len(self.tokenizer)} tokens in the vocabulary")

