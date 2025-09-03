import os
import argparse
import json
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
try:
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    _SKLEARN_OK = True
except Exception:
    _SKLEARN_OK = False

def load_labels(df, label_col):
    labels = sorted(df[label_col].dropna().unique().tolist())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    return labels, label2id, id2label

def map_labels(series, label2id):
    unknown = set(series.unique()) - set(label2id.keys())
    if unknown:
        raise ValueError(f"Unknown labels in data: {unknown}")
    return series.map(label2id).astype(int)

def prepare_dataset(parquet_path: str, text_col: str, label_col: str, label2id):
    df = pd.read_parquet(parquet_path)
    df = df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"})
    df["label"] = map_labels(df["label"], label2id)
    df["text"] = df["text"].astype(str)
    return Dataset.from_pandas(df, preserve_index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    if _SKLEARN_OK:
        return {
            "accuracy": accuracy_score(labels, preds),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }
    return {"accuracy": float((preds == labels).mean())}

def main():
    parser = argparse.ArgumentParser(description="Fine-tune a BERT classifier using train/test Parquet splits.")
    parser.add_argument("--train_pq", type=str, required=True, help="Path to training Parquet file")
    parser.add_argument("--test_pq", type=str, required=True, help="Path to test Parquet file")
    parser.add_argument("--label_col", type=str, required=True, help="Label column to classify")
    parser.add_argument("--text_col", type=str, default="description", help="Text column name")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="HF model name")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for output")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train_bs", type=int, default=16)
    parser.add_argument("--eval_bs", type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading training set from {args.train_pq}")
    train_df = pd.read_parquet(args.train_pq)
    labels, label2id, id2label = load_labels(train_df, args.label_col)
    print(f"Using label column '{args.label_col}' with {len(labels)} unique labels: {labels}")

    train_ds = prepare_dataset(args.train_pq, args.text_col, args.label_col, label2id)
    eval_ds = prepare_dataset(args.test_pq, args.text_col, args.label_col, label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    num_labels = len(labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    data_collator = DataCollatorWithPadding(tokenizer)
    fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        eval_strategy="steps",
        logging_steps=20,
        save_total_limit=2,
        warmup_ratio=0.06,
        weight_decay=0.01,
        fp16=fp16,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro" if _SKLEARN_OK else "accuracy",
        greater_is_better=True,
        seed=42,
        report_to="none"
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=256)

    train_ds_tok = train_ds.map(tokenize, batched=True)
    eval_ds_tok = eval_ds.map(tokenize, batched=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds_tok,
        eval_dataset=eval_ds_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("Starting fine-tuning...")
    trainer.train()

    if _SKLEARN_OK:
        print("Test set classification report:")
        results = trainer.predict(eval_ds_tok)
        y_true, y_pred = results.label_ids, results.predictions.argmax(-1)
        print(classification_report(
            y_true,
            y_pred,
            target_names=[id2label[i] for i in range(num_labels)],
            digits=4
        ))

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "labels.json"), "w") as f:
        json.dump({"labels": labels}, f)
    print(f"Model and config saved to: {args.output_dir}")

if __name__ == "__main__":
    main()