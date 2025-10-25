"""
This module performs K-fold cross-validation training, validation, and testing
of a CNN image classification model using PyTorch. It manages data loading,
training loops, model evaluation, and logging of per-fold metrics.

This is the main file that runs all the other modules, you can either run it
on your personal computer in the CLI:
    python main.py [arguments if needed]

or on the RebelX cluster by running the slurm_run.sh file on the CLI:
    sbatch slurm_run.sh

The training pipeline includes:
    - Reproducible seeding
    - K-fold dataset splitting
    - Model training and validation per epoch
    - Metrics logging and saving for each fold
    - Aggregation of fold-level test results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from tqdm import tqdm

from dataset import CSVDataset
from model import CNNModel
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any

def wbce_pos(outputs: torch.Tensor, labels: torch.Tensor, w_pos: float, w_neg: float) -> torch.Tensor:
    p = (outputs[:, 1]).clamp(1e-7, 1 - 1e-7)  
    y = labels[:, 1]
    # weight                          
    w = torch.where(
        y == 1,
        torch.tensor(w_pos, device=p.device),
        torch.tensor(w_neg, device=p.device),
    )
    return F.binary_cross_entropy(p, y, weight=w)

def focal_pos(
    outputs: torch.Tensor,
    labels: torch.Tensor,
    alpha_pos: float,
    alpha_neg: float,
    gamma: float = 2.0,
) -> torch.Tensor:
    # not exactly but close to 0 to 1
    p = (outputs[:, 1]).clamp(1e-7, 1 - 1e-7)   
    y = labels[:, 1]                          
    # alpha 
    alpha = torch.where(
        y == 1,
        torch.as_tensor(alpha_pos, device=p.device, dtype=p.dtype),
        torch.as_tensor(alpha_neg, device=p.device, dtype=p.dtype),
    )
   # focal loss
    p_t = torch.where(y == 1, p, 1 - p)
    loss = -alpha * (1.0 - p_t).pow(gamma) * torch.log(p_t)
    return loss.mean()

def run_training(args: Any) -> None:
    """
    Execute K-fold cross-validation training and evaluation.

    Args:
        args: Object containing configuration parameters such as dataset directories and image settings.        
    Returns:
        None. Results and logs are written to disk.
    """
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load dataset
    csv_file_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(csv_file_path)
    os.makedirs(os.path.join(args.output_dir, args.version, "logs"), exist_ok=True)

    # Map tumor type to binary label
    df["tumor"] = df["tumor_type"].map({"Benign": 0, "Malignant": 1})

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    fold_results = []

    for fold, (i_train, i_test) in enumerate(kf.split(df)):
        train_df = df.iloc[i_train].reset_index(drop=True)
        test_df = df.iloc[i_test].reset_index(drop=True)


        val_count = max(1, int(round(0.2 * len(train_df))))
        if len(train_df) - val_count < 1:
            val_count = len(train_df) - 1
        train_df, val_df = train_test_split(
            train_df,
            test_size=val_count,
            shuffle=True,
            )
        
        # Initialize datasets and dataloaders
        train_ds = CSVDataset(args, train_df)
        val_ds = CSVDataset(args, val_df)
        test_ds = CSVDataset(args, test_df)

        train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=1)
        val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=1)
        test_dl = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=1)

        # Initialize model, loss, and optimizer
        model = CNNModel(args).to(DEVICE)
        
        # # wbce_pos --------------------------------------------------------------
        # Npos = int((train_df["tumor"] == 1).sum())
        # Nneg = int((train_df["tumor"] == 0).sum())
        # Ntot = max(1, Npos + Nneg)
        # w_pos = Nneg / Ntot   
        # w_neg = Npos / Ntot   
        # # --------------------------------------------------------------------

        # focal_pos --------------------------------------------------------------
        Npos = int((train_df["tumor"] == 1).sum())
        Nneg = int((train_df["tumor"] == 0).sum())
        Ntot = max(1, Npos + Nneg)

        # Class weights (you can reuse the same scheme you used for WBCE)
        alpha_pos = Nneg / Ntot   
        alpha_neg = Npos / Ntot   

        gamma = 1.0               # 1.0, 2.0, 3.0
        # --------------------------------------------------------------------
        
        #criterion = torch.nn.BCELoss()
        #criterion = lambda out, lab: wbce_pos(out, lab, w_pos, w_neg)
        criterion = lambda out, lab: focal_pos(out, lab, alpha_pos, alpha_neg, gamma)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        log_records = []

        # Training loop
        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(model, train_dl, criterion, optimizer, DEVICE)
            val_loss, val_acc, val_prec, val_rec, val_spec, val_f1, val_roc, val_pr = validate(
                model, val_dl, criterion, DEVICE
            )

            log_records.append({
                "fold": fold + 1,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_prec": val_prec,
                "val_rec": val_rec,
                "val_spec": val_spec,
                "val_f1": val_f1,
                "val_roc_auc": val_roc,
                "val_pr_auc": val_pr,
            })

            print(
                f"Epoch [{epoch + 1:02d}/{args.epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}"
            )

        # Save per-fold logs
        log_df = pd.DataFrame(log_records)
        log_path = os.path.join(args.output_dir, args.version, "logs", f"fold_{fold + 1}_log.csv")
        log_df.to_csv(log_path, index=False)

        # Final test evaluation
        test_acc, test_prec, test_rec, test_spec, test_f1, test_roc, test_pr = test(model, test_dl, DEVICE)
        fold_results.append({
            "fold": fold + 1,
            "test_acc": test_acc,
            "test_prec": test_prec,
            "test_rec": test_rec,
            "test_spec": test_spec,
            "test_f1": test_f1,
            "test_roc_auc": test_roc,
            "test_pr_auc": test_pr,
        })

        # Free memory and delete model to prevent leakage over folds
        del model
        torch.cuda.empty_cache()

    # Save aggregated test results
    results_path = os.path.join(args.output_dir, args.version, "testing_results.csv")
    pd.DataFrame(fold_results).to_csv(results_path, index=False)


if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
