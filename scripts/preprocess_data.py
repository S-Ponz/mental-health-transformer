import os, re
import pandas as pd
from sklearn.model_selection import train_test_split

from core.config import Config

def cleanse(s:str):
    cleansed = re.sub(r'[^\x20-\x7E]+', '', s).encode('ascii', errors='ignore').decode('ascii')
    cleansed = re.sub(r"http\S+", "", cleansed)   # remove URLs
    cleansed = re.sub(r"\s+", " ", cleansed)      # normalize whitespace
    return cleansed.lower().replace("\n", " ").strip()


def preprocess_data(reprocess=False):
    # Check if processed data already exists
    if os.path.exists(Config.train_data_path) and os.path.exists(Config.val_data_path) and os.path.exists(Config.test_data_path) and not reprocess:
        print("Processed data already exists. Skipping preprocessing.")
        return

    # Load raw data
    train_val_df = pd.read_csv(Config.train_val_raw_path).rename(columns={"status":Config.label_column})[[Config.text_column, Config.label_column]]
    test_df = pd.read_csv(Config.test_data_raw_path).rename(columns={"status":Config.label_column})[[Config.text_column, Config.label_column]]

    train_val_df['label'] = train_val_df['label'].replace({'stress':'Anxiety','normal':'Normal','depression':'Depression','suicide':'Suicidal'})
    test_df['label'] = test_df['label'].replace({'stress':'Anxiety','normal':'Normal','depression':'Depression','suicide':'Suicidal'})

    #Encode labels
    label_map = Config.label_map
    train_val_df['label'] = train_val_df['label'].map(label_map)
    test_df['label'] = test_df['label'].map(label_map)

    # Cleanse
    train_val_df[Config.text_column] = train_val_df[Config.text_column].apply(cleanse)
    train_val_df[Config.text_column] = train_val_df[Config.text_column].fillna("")
    train_val_df[Config.text_column] = train_val_df[Config.text_column].astype(str)
    train_val_df = train_val_df[train_val_df[Config.text_column].str.strip() != ""]    

    test_df[Config.text_column] = test_df[Config.text_column].apply(cleanse)
    test_df[Config.text_column] = test_df[Config.text_column].fillna("")
    test_df[Config.text_column] = test_df[Config.text_column].astype(str)
    test_df = test_df[test_df[Config.text_column].str.strip() != ""]      

    # Split train/val
    train_df, val_df = train_test_split(train_val_df, test_size=Config.train_val_split, random_state=Config.seed, stratify=train_val_df[Config.label_column])

    # Save processed data
    train_df.to_csv(Config.train_data_path, index=False)
    val_df.to_csv(Config.val_data_path, index=False)
    test_df.to_csv(Config.test_data_path, index=False)

    # Save Tokenizer dataset
    train_df[Config.text_column].to_csv(Config.tokenizer_data_path, index=False, header=False)

    print("Data preprocessing completed. Processed files saved to 'data/processed/' directory.")



