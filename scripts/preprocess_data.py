# =============================================================================
# preprocess_data.py
# -----------------------------------------------------------------------------
# Responsible for turning raw CSV data into clean, model-ready datasets.
#
# Two main functions:
#   cleanse(s)        - Sanitizes a single string: strips non-ASCII characters,
#                       removes URLs, normalizes whitespace, and lowercases.
#   cleanse_data(df)  - Applies cleanse() to a full DataFrame, standardizes
#                       column names, drops nulls/empty rows, normalizes label
#                       variants (e.g. "stress" -> "Anxiety"), and encodes
#                       string labels to integer indices via Config.label_map.
#   preprocess_data() - Orchestrates the full pipeline: reads the raw train/val
#                       and test CSVs, calls cleanse_data(), performs a
#                       stratified train/val split, and writes the three
#                       processed CSVs plus a plain-text file used to train
#                       the tokenizer. Skips re-processing if outputs already
#                       exist unless reprocess=True is passed.
# =============================================================================

import os, re
import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import Config

def cleanse(s:str):
    cleansed = re.sub(r'[^\x20-\x7E]+', '', s).encode('ascii', errors='ignore').decode('ascii')
    cleansed = re.sub(r"http\S+", "", cleansed)   # remove URLs
    cleansed = re.sub(r"\s+", " ", cleansed)      # normalize whitespace
    return cleansed.lower().replace("\n", " ").strip()

def cleanse_data(data:pd.DataFrame, text_column:str="text", label_column:str="label"):

    df = data.copy()

    # Load raw data
    df = df.rename(columns={text_column:Config.text_column, label_column:Config.label_column})[[Config.text_column, Config.label_column]]
    df.dropna(inplace=True)

    df[Config.label_column] = df[Config.label_column].replace({'stress':'Anxiety','normal':'Normal','depression':'Depression','suicide':'Suicidal'})
    
    #Encode labels
    label_map = Config.label_map
    df[Config.label_column] = df[Config.label_column].map(label_map)

    # Cleanse
    df[Config.text_column] = df[Config.text_column].apply(cleanse)
    df[Config.text_column] = df[Config.text_column].fillna("")
    df[Config.text_column] = df[Config.text_column].astype(str)
    df = df[df[Config.text_column].str.strip() != ""]  

    return df

def preprocess_data(reprocess=False):
    # Check if processed data already exists
    if os.path.exists(Config.train_data_path) and os.path.exists(Config.val_data_path) and os.path.exists(Config.test_data_path) and not reprocess:
        print("Processed data already exists. Skipping preprocessing.")
        return
    
    df = pd.read_csv(Config.train_val_raw_path)
    train_val_df = cleanse_data(df, text_column="text", label_column="status")

    df = pd.read_csv(Config.test_data_raw_path)
    test_df = cleanse_data(df, text_column="text", label_column="status")

    # Split train/val
    train_df, val_df = train_test_split(train_val_df, test_size=Config.train_val_split, random_state=Config.seed, stratify=train_val_df[Config.label_column])

    # Save processed data
    train_df.to_csv(Config.train_data_path, index=False)
    val_df.to_csv(Config.val_data_path, index=False)
    test_df.to_csv(Config.test_data_path, index=False)

    # Save Tokenizer dataset
    train_df[Config.text_column].to_csv(Config.tokenizer_data_path, index=False, header=False)

    print("Data preprocessing completed. Processed files saved to 'data/processed/' directory.")



