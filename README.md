# mental-health-transformer

Transformer-based NLP model for detecting mental health signals in text.

The model classifies sentences or social media posts into four categories:

- **Normal**
- **Anxiety**
- **Depression**
- **Suicidal**

This project demonstrates how transformer architectures can be applied to mental health text classification tasks.

---

## Project Overview

Mental health signals often appear in written language across social media posts, online forums, and personal messages. Detecting these signals early can support research into automated mental health screening tools.

This repository implements a transformer-based model trained on multiple publicly available datasets to classify mental health related text.

---

## Dataset Sources

### The training data is compiled from a single public dataset:

- Mental Health Text Classification Dataset [HuggingFace](https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset)
Mukherjee, P. (2025). Mental Health Text Classification Dataset (4‑Class) [Dataset]. Hugging Face Hub. https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset.

### This dataset contains user-generated text labeled with mental health categories and is derived from three original datasets.

#### Original datasets

Komati, N. Suicide and Depression Detection [Dataset]. Kaggle.
https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch

Sarkar, S. Sentiment Analysis for Mental Health [Dataset]. Kaggle.
https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health

Murarka, A., Radhakrishnan, B., & Ravichandran, S. (2021). Detection and Classification of Mental Illnesses on Social Media using RoBERTa [Dataset and code]. GitHub.
https://github.com/amurark/mental-health-detection

---

## Label Categories

| Label | Description |
|------|-------------|
| Normal | No strong mental health indicators |
| Anxiety | Expressions of worry, panic, or stress |
| Depression | Expressions of sadness, hopelessness, or emotional exhaustion |
| Suicidal | Indications of suicidal ideation or self-harm thoughts |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/S-Ponz/mental-health-transformer.git
cd mental-health-transformer
```

## Install dependencies
```bash
pip install -r requirements.txt
```

---

## Training the Model
Run the training script:
```bash
python -m core.train --epochs=1
```

---

## Evaluating the Model
Run Evaluation:
```bash
python -m core.evaluate --file_path="data/raw/mental_health_combined_test.csv" --text_column="text" --label_column="status"

python -m core.evaluate --file_path="data/raw/mental_heath_unbanlanced.csv" --text_column="text" --label_column="status"
```
Evaluation outputs include:
* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix

---

## Running Inference
Examples:
```bash
python -m core.inference --text="I feel like nothing matters anymore and I don't want to wake up."

python -m core.inference --text="It would be kinder to everyone to just end it, why can’t I do it?I know I would be better off dead."

python -m core.inference --text="i've been feeling anxious today, but it's always a good reminder for everyone. the fear and anxious feelings and symptoms won't last forever. you are loved and cared for, and it's going to be okay. &lt;3"

python -m core.inference --text="Keep hurting me , it's fine I'm fine it's OKAY, you aren't the first one to hurt me you aren't special I'm sorry :3 Sorry :3"
```

Example output (Category, Probability):
[('Suicidal', 0.7206500768661499)]

---
