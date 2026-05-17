from pathlib import Path
import yaml
import torch
import networkx as nx
import numpy as np
from sklearn.metrics import (jaccard_score, hamming_loss, precision_score, 
                             recall_score, f1_score, roc_auc_score)


HERE = Path.cwd()  
DATASETS = HERE / "datasets"
CONFIG = HERE / "config.yaml"
ADJ_MATRIX = DATASETS / "synapse_count_matrices.xlsx"

def load_config(CONFIG):
    with open(CONFIG, 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config(CONFIG)


# =====================================================================
# Training metric computation
# =====================================================================
def compute_train_metrics(A_target, A_model, soft_A_model, config=None):
    # Handle shape mismatch
    if A_target.shape != A_model.shape:
        print(f"Shape mismatch: target {A_target.shape}, model {A_model.shape}")
        return None
    
    # Flatten ONLY valid candidate edges (exclude diagonal self-loops)
    mask = ~np.eye(A_target.shape[0], dtype=bool)
    y_target = A_target.astype(int)[mask].ravel()
    y_model  = A_model.astype(int)[mask].ravel()
    y_soft_model = soft_A_model.astype(float)[mask].ravel()

    # 1) Hamming 
    hamming_norm = hamming_loss(y_target, y_model)
    n_entries = y_target.size
    hamming_raw = int(hamming_norm * n_entries)
    
    adjacency_similarity = 1.0 - hamming_norm

    # 2) Jaccard over edges
    edge_jacc = jaccard_score(y_target, y_model, labels=[0,1], average="binary", zero_division=0)

    # 3) precision, recall, F1
    precision = precision_score(y_target, y_model, labels=[0,1], average="binary", zero_division=0)
    recall = recall_score(y_target, y_model, labels=[0,1], average="binary", zero_division=0)
    f1 = f1_score(y_target, y_model, labels=[0,1], average="binary", zero_division=0)

    # 4) AUROC 
    auroc = roc_auc_score(y_target, y_soft_model)

    # 5) Soft edge balanced loss
    soft_edge_balanced_loss = soft_edge_balanced_loss_binary_target(
        A_target,
        soft_A_model,
        config,
    )

    return {
        "hamming_norm": hamming_norm,
        "adjacency_similarity": adjacency_similarity,
        "edge_jaccard": edge_jacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "soft_edge_balanced_loss": soft_edge_balanced_loss,
    }



# =====================================================================
# Soft edge balanced loss
# =====================================================================
def soft_edge_balanced_loss_binary_target(A_target, soft_A_model, config):
    if A_target.shape != soft_A_model.shape:
        raise ValueError(f"Shape mismatch: A_target {A_target.shape} vs soft_A_model {soft_A_model.shape}")

    A = A_target.astype(np.float64)
    P = soft_A_model.astype(np.float64)

    # exclude diagonal (same convention as compute_train_metrics)
    mask = ~np.eye(A.shape[0], dtype=bool)
    A = A[mask]
    P = P[mask]

    # clip only for numerical sanity; does NOT silently change logic in any meaningful way
    P = np.clip(P, 0.0, 1.0)

    soft_fn = np.sum(A * (1.0 - P))
    soft_fp = np.sum((1.0 - A) * P)

    lam_fp = config["lam_fp"]
    n_cells = float(A.size)  # off-diagonal count after the mask above

    return float((soft_fn + lam_fp * soft_fp) / n_cells)
