# =============================================================================
# evaluate.py
# -----------------------------------------------------------------------------
# Runs a trained model against a labelled CSV dataset and produces both
# quantitative metrics and visual plots.
#
# run_evaluation(file_path) workflow:
#   1. Load the saved tokenizer and model checkpoint.
#   2. Read and cleanse the CSV, build an inference DataLoader.
#   3. Pass all batches through the model (no gradients), collecting
#      predictions, max softmax probabilities, and true labels.
#   4. Compute overall accuracy, average cross-entropy loss, and a full
#      per-class classification report (precision, recall, F1, support).
#   5. Save all metrics to outputs/metrics/<filename>.json.
#   6. Generate and save five Plotly charts to outputs/plots/:
#        - Training loss & accuracy curves (if train_stats.json exists)
#        - Confusion matrix
#        - Per-class precision / recall / F1 bar chart
#        - Confidence distribution violin plot per class
#        - Example predictions table (correct & incorrect samples)
#        - Accuracy and sample retention vs. confidence threshold
#
# Run directly (python -m core.evaluate --file_path <csv>) to evaluate from
# the command line.
# =============================================================================

import os
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from core.dataset import get_inference_dataloader
from core.tokenizer import Tokenizer
from core.model import TransformerClassifier
from core.config import Config
from scripts.preprocess_data import cleanse_data
from scripts.plotting import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_probability_distribution,
    plot_example_predictions,
    plot_threshold_analysis,
)


PLOT_DIR = Config.plot_output_dir
TRAIN_STATS_PATH = f"{Config.log_save_path}/train_stats.json"
METRICS_SAVE_DIR = Config.metrics_output_dir

def run_evaluation(file_path:str, text_column:str=Config.text_column, label_column=Config.label_column):

    tokenizer = Tokenizer().tokenizer
    if tokenizer is None:
        raise Exception(f"No tokenizer found in {Config.tokenizer_path}!")

    if not os.path.exists(Config.model_save_path):
        raise Exception(f"No trained model found in {Config.model_save_path}!")

    df = pd.read_csv(file_path)
    
    df = cleanse_data(df, text_column=text_column, label_column=label_column)
    
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

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(Config.device)
            attention_mask = batch["attention_mask"].to(Config.device)
            labels = batch["label"].to(Config.device)

            logits = model(input_ids, attention_mask=attention_mask)

            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            probs = torch.softmax(logits, dim=1)
            max_result = torch.max(probs, dim=1)            

            all_preds.extend(max_result.indices.detach().cpu().numpy())
            all_probs.extend(max_result.values.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())            

    avg_loss = total_loss / len(loader.dataset) # type: ignore
    acc = accuracy_score(all_labels, all_preds)

    target_names = [Config.inv_label_map[l] for l in set(all_preds)]

    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=target_names,
        output_dict=True
    )    

    results = {
        'N':df.shape[0],
        'loss':avg_loss,
        'accuracy':acc,
        'classification_report':class_report,
        'results':{'labels':np.array(all_labels, dtype=float).tolist(), 'preds':np.array(all_preds, dtype=float).tolist(), 'probabilities':np.array(all_probs, dtype=float).tolist()}
    }

    file_name = os.path.split(file_path)[-1].replace('.csv','')
    save_path = f"{METRICS_SAVE_DIR}/{file_name}.json"

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=3)

    # --- Plots ---
    os.makedirs(PLOT_DIR, exist_ok=True)

    # Training curves (only if the stats file exists)
    if os.path.exists(TRAIN_STATS_PATH):
        plot_training_curves(TRAIN_STATS_PATH, PLOT_DIR)

    texts = df[text_column].astype(str).tolist()

    plot_confusion_matrix(all_labels, all_preds, Config.inv_label_map, PLOT_DIR, file_name)
    plot_per_class_metrics(class_report, PLOT_DIR, file_name) # type: ignore
    plot_probability_distribution(all_preds, all_probs, Config.inv_label_map, PLOT_DIR, file_name)
    plot_example_predictions(texts, all_labels, all_preds, all_probs, Config.inv_label_map, PLOT_DIR, file_name)
    plot_threshold_analysis(all_labels, all_preds, all_probs, Config.inv_label_map, PLOT_DIR, file_name)

    return results, save_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A script that runs text from a csv file through a trained mental health transformer to be evaluated.")
    parser.add_argument("--file_path", type=str, help="The file path of text to be ran through the transformer for evaluation.")
    parser.add_argument("--text_column", type=str, help="columns of the texts")
    parser.add_argument("--label_column", type=str, help="column of the labels")
    
    args = parser.parse_args()

    results, save_path = run_evaluation(args.file_path, args.text_column, args.label_column)

    print(f"Results saved to: {save_path}")
