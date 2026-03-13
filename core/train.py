import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score

from scripts import preprocess_data
from core.tokenizer import Tokenizer
from core.model import TransformerClassifier
from core.config import Config
from core.dataset import get_dataloaders

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

        logits = model(input_ids, attention_mask=attention_mask)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, np.array(all_labels), np.array(all_preds)



def train(model, train_loader, val_loader, optimizer, criterion, device=Config.device, num_epochs=Config.num_epochs):

    best_val_acc = 0.0
    for epoch in range(num_epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_path)
            print("Best model saved!")


if __name__ == "__main__":

    print("Preprocessing data...")
    preprocess_data.preprocess_data(reprocess=True)

    print("Training tokenizer...")
    tokenizer = Tokenizer().tokenizer
    if tokenizer is None:
        Tokenizer().train()
        tokenizer = Tokenizer().tokenizer
        if tokenizer is None:
            raise ValueError("Failed to train tokenizer.")

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
        dropout=0.1,
        layers=Config.num_layers
    ).to(Config.device)


    print("Preparing data loaders...")
    train_loader, val_loader = get_dataloaders(tokenizer)
                    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate)

    print("Starting training...")
    train(model, train_loader, val_loader, optimizer, criterion)

