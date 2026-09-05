"""
Train a lightweight harm / sexism classifier for raw-text platforms (Reddit, Telegram).

Uses EDOS + Kaggle women harassment as training data; saves model to models/harm_classifier/.

Run (after placing edos CSV in data/raw/ or project root):
  python src/train_harm_classifier.py
  python src/classifier.py --input data/scraped/telegram_live.csv --use-model
"""

from __future__ import annotations

import os
import sys

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))

MODEL_DIR = "models/harm_classifier"


def load_training_frames() -> pd.DataFrame:
    frames = []
    edos_paths = [
        "data/raw/edos_labelled_aggregated.csv",
        "../edos_labelled_aggregated.csv",
    ]
    for p in edos_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            text_col = next((c for c in df.columns if "text" in c.lower()), None)
            label_col = next((c for c in df.columns if "sexist" in c.lower()), None)
            if text_col and label_col:
                frames.append(
                    pd.DataFrame(
                        {
                            "text": df[text_col].astype(str),
                            "label": (df[label_col].astype(str).str.lower() == "sexist").astype(int),
                        }
                    )
                )
            break

    wh_path = "data/raw/women_harassment.csv"
    if os.path.exists(wh_path):
        df = pd.read_csv(wh_path)
        tc = next((c for c in df.columns if "text" in c.lower() or "content" in c.lower()), None)
        lc = next((c for c in df.columns if "label" in c.lower() or "class" in c.lower()), None)
        if tc:
            frames.append(
                pd.DataFrame(
                    {
                        "text": df[tc].astype(str),
                        "label": pd.to_numeric(df[lc], errors="coerce").fillna(0).astype(int)
                        if lc
                        else 1,
                    }
                )
            )

    if not frames:
        raise FileNotFoundError("Need edos_labelled_aggregated.csv or women_harassment.csv for training.")

    data = pd.concat(frames, ignore_index=True)
    data = data[data["text"].str.len() > 10].drop_duplicates(subset=["text"])
    return data


def main():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
        from datasets import Dataset
    except ImportError:
        print("Install: pip install transformers datasets torch accelerate")
        return

    data = load_training_frames()
    train_df, test_df = train_test_split(data, test_size=0.15, random_state=42, stratify=data["label"])

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_ds = Dataset.from_pandas(train_df).map(tok, batched=True)
    test_ds = Dataset.from_pandas(test_df).map(tok, batched=True)

    args = TrainingArguments(
        output_dir=MODEL_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    preds = trainer.predict(test_ds)
    pred_labels = preds.predictions.argmax(axis=1)
    print(classification_report(test_df["label"], pred_labels, target_names=["safe", "harmful"]))
    print(f"Model saved → {MODEL_DIR}")


if __name__ == "__main__":
    main()
