# mental-health-transformer

Transformer-based NLP model for detecting mental health signals in text.

The model classifies sentences or social media posts into four categories:

- **Normal**
- **Anxiety**
- **Depression**
- **Suicidal Ideation**

This project demonstrates how transformer architectures can be applied to mental health text classification tasks.

---

## Project Overview

Mental health signals often appear in written language across social media posts, online forums, and personal messages. Detecting these signals early can support research into automated mental health screening tools.

This repository implements a transformer-based model trained on multiple publicly available datasets to classify mental health related text.

---

## Dataset Sources

The training data is compiled from a single public dataset:

- Mental Health Text Classification Dataset (HuggingFace)

This dataset contains user-generated text labeled with mental health categories.

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
python -m core.train
```

---

## Evaluating the Model
Run Evaluation:
```bash
python -m core.evaluate
```
Evaluation outputs include:
* Accuracy
* Precision
* Recall
* F1 score
* Confusion matrix

---

## Running Inference
Example:
```bash
python -m core.inference "I feel like nothing matters anymore and I don't want to wake up."
```

Example output:
Prediction: Depression
Confidence: 0.87

---
