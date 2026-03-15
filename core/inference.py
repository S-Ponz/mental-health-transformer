# =============================================================================
# inference.py
# -----------------------------------------------------------------------------
# Provides a lightweight interface for running the trained model on new,
# unlabelled text at inference time (no ground-truth labels required).
#
# run_inference(texts, labels) workflow:
#   1. Load the saved tokenizer and model checkpoint.
#   2. Accept a list of raw text strings (and optionally true labels; defaults
#      to "Normal" placeholders so the DataLoader format is satisfied).
#   3. Cleanse and tokenize the texts, then pass them through the model.
#   4. Return a list of (predicted_label, confidence) tuples — one per input.
#
# Intended use-cases:
#   - Interactive demos or APIs that classify a single piece of text.
#   - Batch scoring of new data where labels are not yet known.
#
# Run directly (python -m core.inference --text "...") to classify a single
# string from the command line.
# =============================================================================

import os
import torch
import argparse
import pandas as pd

from core.dataset import get_inference_dataloader
from core.tokenizer import Tokenizer
from core.model import TransformerClassifier
from core.config import Config
from scripts.preprocess_data import cleanse_data

def run_inference(texts:list[str], labels:list[str]|None=None):


    tokenizer = Tokenizer().tokenizer
    if tokenizer is None:
        raise Exception(f"No tokenizer found in {Config.tokenizer_path}!")

    if not os.path.exists(Config.model_save_path):
        raise Exception(f"No trained model found in {Config.model_save_path}!")
    
    _labels = labels if labels is not None else ['Normal']*len(texts)

    if len(texts) == 1:
        df = pd.DataFrame({Config.text_column:texts, Config.label_column:_labels}, index=[0])
    else:
        df = pd.DataFrame({Config.text_column:texts, Config.label_column:_labels})

    df = cleanse_data(df, text_column=Config.text_column, label_column=Config.label_column)
    
    loader = get_inference_dataloader(df, tokenizer, batch_size=Config.batch_size, max_len=Config.max_length)

    vocab_size = len(tokenizer)
    pad_idx = tokenizer.pad_token_id

    model = TransformerClassifier(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        num_positions=Config.max_length,
        d_model=Config.d_model,
        d_internal=Config.d_internal,
        num_heads=Config.num_heads,
        num_classes=Config.num_classes,
        dropout=0.0,
        layers=Config.num_layers
    ).to(Config.device)

    model.load_state_dict(torch.load(Config.model_save_path, map_location=Config.device)) 
    model.eval()


    results = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)

            logits = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1)
            max_result = torch.max(probs, dim=1)

            results.extend([(Config.inv_label_map[p], prob) for p,prob in zip(max_result.indices.tolist(), max_result.values.tolist())])

    return results



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that runs text through a trained mental health transformer to be categorized.")
    parser.add_argument("--text", type=str, help="The test to run through the transformer.")
    
    args = parser.parse_args()


    results = run_inference([args.text])

    print(f"{results}")


