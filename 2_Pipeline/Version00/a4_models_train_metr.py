import numpy as np
from sklearn.metrics import (jaccard_score, hamming_loss, precision_score, recall_score, f1_score)

######################
# Training objectives
######################

def train_metrics(A_target, A_model, verbose=False):
    if A_target.shape != A_model.shape:
        raise ValueError(f"Shape mismatch: target {A_target.shape}, model {A_model.shape}")

    # Flatten adjacency matrices to 1D label vectors
    y_true = A_target.astype(int).ravel()
    y_pred = A_model.astype(int).ravel()

    if verbose:
        print(A_target)
        print(y_true)
        
        print(A_model)
        print(y_pred)
        

    # 1) Hamming (matrix-level)
    hamming_norm = hamming_loss(y_true, y_pred)
    n_entries = y_true.size
    hamming_raw = int(hamming_norm * n_entries)
    adjacency_similarity = 1.0 - hamming_norm

    # 2) Jaccard over edges
    edge_jacc = jaccard_score(y_true, y_pred)

    # 3) Other edge metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    return {
        "hamming_raw": hamming_raw,
        "hamming_norm": hamming_norm,
        "adjacency_similarity": adjacency_similarity,
        "edge_jaccard": edge_jacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_entries": n_entries,
    }





##########################
# Subgraph example
##########################

def print_subgraph(A_target, A_model, neurons, idx=None):
    if idx is None:
        idx = [0, 1, 2, 3, 4, 5]

    labels = [neurons[i] for i in idx]

    sub_target = A_target[np.ix_(idx, idx)]
    sub_model = A_model[np.ix_(idx, idx)]

    # Print adjacency matrices with labels
    def print_matrix(name, M):
        print(f"\n{name} adjacency (subgraph):")
        print("           " + "  ".join(f"{lab:>8}" for lab in labels))
        for row_label, row in zip(labels, M):
            row_str = "  ".join(str(int(v)) for v in row)
            print(f"{row_label:>8}  {row_str}")

    print_matrix("TARGET", sub_target)
    print_matrix("MODEL ", sub_model)

    # Compute metrics on the subgraph 
    metrics = train_metrics(sub_target, sub_model)

    print("\n  [subgraph] metrics (over flattened adjacency):")
    print(f"    n_entries            = {metrics['n_entries']}")
    print(f"    hamming_raw          = {metrics['hamming_raw']} / {metrics['n_entries']}")
    print(f"    hamming_norm         = {metrics['hamming_norm']:.4f}")
    print(f"    adjacency_similarity = {metrics['adjacency_similarity']:.4f}")
    print(f"    edge_jaccard         = {metrics['edge_jaccard']:.4f}")
    print(f"    precision            = {metrics['precision']:.4f}")
    print(f"    recall               = {metrics['recall']:.4f}")
    print(f"    f1                   = {metrics['f1']:.4f}")