import os
import json
import glob
import random
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

import torch


def seed_everything(seed=42):
    """
    Set all random seeds for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def read_json_file(path):
    """
    Read line-delimited JSON into a pandas DataFrame.
    """
    with open(path) as f:
        lines = f.read().splitlines()
    return pd.DataFrame([json.loads(line) for line in lines])


def load_json_dataset(directory_pattern):
    """
    Load multiple JSON files into a single DataFrame.
    """
    df_list = [read_json_file(path) for path in glob.glob(directory_pattern)]
    return pd.concat(df_list, ignore_index=True)


def clean_data(df):
    """
    Remove rows with invalid or noisy labels and explanations.
    """
    invalid_labels = [-1000, -5000, -1]
    df = df[df['explanation'] != "Invalid response format"]
    df = df[~df['predicted_label'].isin(invalid_labels)]
    return df[['label', 'explanation', 'ans0', 'ans1', 'ans2']]


def preprocess_function(example, tokenizer, max_length=4096):
    """
    Tokenize and format the input text for classification.
    """
    text = (
        f"Explanataion:\n{example['explanation']} [SEP] Choices \n"
        f"0: {example['ans0']} \n"
        f"1: {example['ans1']} \n"
        f"2: {example['ans2']} \n"
    )
    return tokenizer(text, max_length=max_length, padding=True, truncation=True)


def compute_metrics(pred):
    """
    Compute accuracy, precision, recall, and F1 for evaluation.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def tokenize_datasets(ds_train, ds_valid, tokenizer, max_length=4096):
    """
    Apply tokenization to both training and validation datasets.
    """
    return (
        ds_train.map(lambda x: preprocess_function(x, tokenizer, max_length)),
        ds_valid.map(lambda x: preprocess_function(x, tokenizer, max_length))
    )


def get_trainer(model, tokenizer, ds_train, ds_valid, output_dir, batch_size=4, epochs=5):
    """
    Initialize the HuggingFace Trainer with training arguments.
    """
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-6,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="none",
        save_total_limit=2,
        lr_scheduler_type="linear",
        optim="adamw_torch"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_valid,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    return trainer


def main(
    model_name='microsoft/deberta-v3-large',
    train_path='../thoughts/gemma/split/train/*',
    val_path='../thoughts/gemma/split/val/*',
    output_dir='/mnt/drive/VLCQ/biased_thoughts_output',
    seed=1,
    num_labels=3,
    batch_size=4,
    epochs=5,
    max_length=4096
):
    """
    Full training pipeline for explanation-based bias classification.
    """
    seed_everything(seed)

    # Load and clean datasets
    df_train = clean_data(load_json_dataset(train_path))
    df_valid = clean_data(load_json_dataset(val_path))

    # Convert to HuggingFace Datasets
    ds_train = Dataset.from_pandas(df_train)
    ds_valid = Dataset.from_pandas(df_valid)

    # Load tokenizer and tokenize datasets
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds_train_enc, ds_valid_enc = tokenize_datasets(ds_train, ds_valid, tokenizer, max_length)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Initialize Trainer and train
    trainer = get_trainer(model, tokenizer, ds_train_enc, ds_valid_enc, output_dir, batch_size, epochs)
    trainer.train()


if __name__ == "__main__":
    main()