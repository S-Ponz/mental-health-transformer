import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from core.config import Config

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=Config.max_length):
        self.texts = df[Config.text_column].astype(str).tolist()
        self.labels = df[Config.label_column].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            padding=Config.padding,
            max_length=self.max_len,
            return_attention_mask=True,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }



def get_dataloaders(tokenizer, batch_size=Config.batch_size, max_len=Config.max_length):

    train_df = pd.read_csv(Config.train_data_path)
    val_df = pd.read_csv(Config.val_data_path)

    train_dataset = TextDataset(train_df, tokenizer, max_len=max_len)
    val_dataset = TextDataset(val_df, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader