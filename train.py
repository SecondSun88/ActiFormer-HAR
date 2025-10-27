# train.py
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import time

# Import your models
from ActiFormer import ActiFormer
from baselines import (CNN_Baseline, Transformer_Baseline, Performer_Baseline,
                       Informer_Baseline, LiteTransformer_Baseline)


# --- DATASET PREPROCESSING (Simplified) ---
# NOTE: Place your actual data loading/preprocessing functions here.
def load_dataset(name, path):
    print(f"Loading {name} dataset...")
    # This is a placeholder. Replace with your actual data loading logic.
    if name == 'UCI-HAR':
        return np.random.rand(7352, 128, 6), np.random.randint(0, 6, 7352), \
               np.random.rand(2947, 128, 6), np.random.randint(0, 6, 2947)
    if name == 'PAMAP2':
        return np.random.rand(17133, 171, 52), np.random.randint(0, 12, 17133), \
               np.random.rand(4284, 171, 52), np.random.randint(0, 12, 4284)
    else:  # Default for UniMiB, OPPORTUNITY
        return np.random.rand(10000, 150, 3), np.random.randint(0, 17, 10000), \
               np.random.rand(2500, 150, 3), np.random.randint(0, 17, 2500)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data Loading and Preparation ---
    x_train_raw, y_train_raw, _, _ = load_dataset(args.dataset, "path/to/data")
    x_train_raw = x_train_raw.transpose(0, 2, 1)

    mean = np.mean(x_train_raw, axis=(0, 2), keepdims=True)
    std = np.std(x_train_raw, axis=(0, 2), keepdims=True)
    x_train_raw = (x_train_raw - mean) / (std + 1e-7)

    x_train, x_test, y_train, y_test = train_test_split(x_train_raw, y_train_raw, test_size=0.2, random_state=42,
                                                        stratify=y_train_raw)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.125, random_state=42,
                                                      stratify=y_train)

    X_train, y_train = torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long()
    X_val, y_val = torch.from_numpy(x_val).float(), torch.from_numpy(y_val).long()
    X_test, y_test = torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch, shuffle=False)

    # --- Model Selection ---
    input_shape = X_train.shape
    num_classes = len(np.unique(y_train_raw))
    model_dict = {
        'ActiFormer': ActiFormer,
        'CNN': CNN_Baseline,
        'Transformer': Transformer_Baseline,
        'Performer': Performer_Baseline,
        'Informer': Informer_Baseline,
        'LiteTransformer': LiteTransformer_Baseline
    }
    model = model_dict[args.model](input_shape, num_classes).to(device)
    print(f"Training model: {args.model} on dataset: {args.dataset}")

    # --- Training ---
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(args.epoch):
        model.train()
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch + 1}/{args.epoch}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'{args.model}_{args.dataset}_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 30:
            print("Early stopping triggered.")
            break

    # --- Testing ---
    model.load_state_dict(torch.load(f'{args.model}_{args.dataset}_best.pth'))
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    print(f"\n--- Final Test Results for {args.model} on {args.dataset} ---")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1-Score: {test_f1:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ActiFormer Training and Evaluation')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['UCI-HAR', 'UniMiB-SHAR', 'PAMAP2', 'OPPORTUNITY'])
    parser.add_argument('--model', type=str, required=True,
                        choices=['ActiFormer', 'CNN', 'Transformer', 'Performer', 'Informer', 'LiteTransformer'])
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    main(args)