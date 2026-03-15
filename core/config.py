from dataclasses import dataclass, field
import torch


@dataclass
class Config:

    seed: int = 45

    # -----------------------------
    # Dataset
    # -----------------------------
    train_val_split: float = 0.2
    train_val_raw_path:str = "data/raw/mental_heath_unbanlanced.csv"
    test_data_raw_path: str = "data/raw/mental_health_combined_test.csv"
    train_data_path: str = "data/processed/train.csv"
    val_data_path: str = "data/processed/val.csv"
    test_data_path: str = "data/processed/test.csv"

    text_column: str = "text"
    label_column: str = "label"

    label_map = {
        "Normal": 0,
        "Anxiety": 1,
        "Depression": 2,
        "Suicidal": 3
    }

    inv_label_map = {v:k for k,v in label_map.items()}

    num_classes: int = 4


    # -----------------------------
    # Tokenizer
    # -----------------------------
    tokenizer_data_path:str = "data/processed/tokenizer_data.txt"
    tokenizer_path: str = "my_tokenizer"
    vocab_size: int = 16000
    min_frequency: int = 20
    special_tokens: tuple = field(default = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"))
    max_length: int = 256
    padding: str = "max_length"
    truncation: bool = True


    # -----------------------------
    # Model
    # -----------------------------
    d_model: int = 128
    d_internal: int = 256
    num_heads: int = 8
    num_layers: int = 2


    # -----------------------------
    # Training
    # -----------------------------
    batch_size: int = 128
    learning_rate: float = 2e-5
    num_epochs: int = 15



    # -----------------------------
    # Hardware
    # -----------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


    # -----------------------------
    # Output
    # -----------------------------
    log_save_path: str = "outputs/logs"
    model_save_path: str = "models/checkpoints/best_model.pt"
    metrics_output_dir: str = "outputs/metrics"
    plot_output_dir: str = "outputs/plots"