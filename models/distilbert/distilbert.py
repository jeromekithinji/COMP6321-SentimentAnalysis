import os
import argparse
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)

# Dataset
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Training and Evaluation
def train_one_epoch(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch in tqdm(data_loader, desc="Train"):
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * input_ids.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    return avg_loss, acc

def eval_one_epoch(model, data_loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Eval"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)
            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)

    avg_loss = total_loss / max(total_examples, 1)
    acc = total_correct / max(total_examples, 1)
    return avg_loss, acc


def main():
    ap = argparse.ArgumentParser(description="Fine-tune DistilBERT for multi-class sentiment.")
    ap.add_argument("--data", required=True, help="Path to CSV with columns cleaned_text,sentiment (or text,label).")
    ap.add_argument("--output", default="./distilbert-multiclass", help="Where to save model/tokenizer.")
    ap.add_argument("--model_name", default="distilbert-base-uncased",
                    help="HF model name or a local directory (useful offline).")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--no_scheduler", action="store_true", help="Disable LR scheduler.")
    args = ap.parse_args()

    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load data
    df = pd.read_csv(args.data)
    # Normalize columns
    if "text" not in df.columns and "cleaned_text" in df.columns:
        df = df.rename(columns={"cleaned_text": "text"})
    if "label" not in df.columns and "sentiment" in df.columns:
        df = df.rename(columns={"sentiment": "label"})

    missing = [c for c in ("text", "label") if c not in df.columns]
    if missing:
        raise ValueError(f"CSV must contain columns 'text' and 'label' (or 'cleaned_text' and 'sentiment'). Missing: {missing}")

    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    num_classes = df["label"].nunique()
    print(f"num_classes = {num_classes}")

    # Split data
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df["label"]
    )
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)}")

    # Tokenizer & datasets
    local_only = os.path.isdir(args.model_name)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_name, local_files_only=local_only)

    train_ds = ReviewDataset(train_df["text"], train_df["label"], tokenizer, args.max_len)
    val_ds   = ReviewDataset(val_df["text"],   val_df["label"],   tokenizer, args.max_len)

    # Dataloaders
    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = DistilBertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_classes, local_files_only=local_only
    ).to(device)

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = None if args.no_scheduler else get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # Train
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  Train loss: {tr_loss:.4f} | Train acc: {tr_acc:.4f}")
        va_loss, va_acc = eval_one_epoch(model, val_loader, device)
        print(f"  Val   loss: {va_loss:.4f} | Val   acc: {va_acc:.4f}")

    
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Saved to:", args.output)

    # Classification report
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    
    uniq = sorted(df["label"].unique().tolist())
    target_names = [str(u) for u in uniq]
    print("\nClassification report (val set):")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))

if __name__ == "__main__":
    main()