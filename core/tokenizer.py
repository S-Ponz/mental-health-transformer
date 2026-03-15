# =============================================================================
# tokenizer.py
# -----------------------------------------------------------------------------
# Wraps a custom WordPiece tokenizer trained on the project's own text corpus.
#
# The Tokenizer class:
#   - On instantiation, attempts to load a previously saved tokenizer from
#     Config.tokenizer_path. If none exists, self.tokenizer is set to None.
#   - train() trains a BertWordPieceTokenizer from scratch on the processed
#     training text (Config.tokenizer_data_path), using the vocabulary size,
#     minimum token frequency, and special tokens defined in Config. The
#     trained vocabulary is saved to disk and reloaded as a BertTokenizerFast
#     for fast batch encoding.
#
# Using a domain-trained tokenizer (rather than a pre-trained one) means the
# vocabulary is tuned to the language patterns found in mental-health text,
# resulting in more efficient tokenization of relevant terms.
# =============================================================================

import os
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast

from core.config import Config

class Tokenizer:
    def __init__(self, tokenizer_path=Config.tokenizer_path):

        self.tokenizer = None
        self.tokenizer_path = tokenizer_path
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

        if not os.path.exists(self.tokenizer_path):
            os.makedirs(self.tokenizer_path, exist_ok=True)

        tokenizer.save_model(self.tokenizer_path)

        self.tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path)

        print(f"{len(self.tokenizer)} tokens in the vocabulary")

      

