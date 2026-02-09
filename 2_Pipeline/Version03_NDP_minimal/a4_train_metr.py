import numpy as np
from sklearn.metrics import (jaccard_score, hamming_loss, precision_score, 
                             recall_score, f1_score)


def compute_train_metrics(A_target, A_model, verbose=True):
    # Handle shape mismatch
    if A_target.shape != A_model.shape:
        if verbose:
            print(f"Shape mismatch: target {A_target.shape}, model {A_model.shape}")
        return None

    # Flatten adjacency matrices to 1D label vectors
    y_target = A_target.astype(int).ravel()
    y_model = A_model.astype(int).ravel()

    # if verbose:
        # print(f"Target adjacency:\n{A_target}")
        # print(f"Target flattened: {y_target}")
        # print(f"Model adjacency:\n{A_model}")
        # print(f"Model flattened: {y_model}")

    # 1) Hamming (matrix-level)
    hamming_norm = hamming_loss(y_target, y_model)
    n_entries = y_target.size
    hamming_raw = int(hamming_norm * n_entries)
    
    # if verbose:
        # print(f"Hamming raw: {hamming_raw}")
        # print(f"n_entries: {n_entries}")

    adjacency_similarity = 1.0 - hamming_norm

    # 2) Jaccard over edges
    edge_jacc = jaccard_score(y_target, y_model, zero_division=0)

    # 3) Other edge metrics
    precision = precision_score(y_target, y_model, zero_division=0)
    recall = recall_score(y_target, y_model, zero_division=0)
    f1 = f1_score(y_target, y_model, zero_division=0)

    return {
        "hamming_norm": hamming_norm,
        "adjacency_similarity": adjacency_similarity,
        "edge_jaccard": edge_jacc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_subgraph(A_target, A_model, neurons, idx=None):
    if idx is None:
        idx = [0, 1, 2, 3, 4, 5]

    labels = [neurons[i] for i in idx]

    sub_target = A_target[np.ix_(idx, idx)]
    sub_model = A_model[np.ix_(idx, idx)]

    def print_matrix(name, M):
        print(f"\n{name} adjacency (subgraph):")
        print("           " + "  ".join(f"{lab:>8}" for lab in labels))
        for row_label, row in zip(labels, M):
            row_str = "  ".join(str(int(v)) for v in row)
            print(f"{row_label:>8}  {row_str}")

    print_matrix("TARGET", sub_target)
    print_matrix("MODEL ", sub_model)

    metrics = compute_train_metrics(sub_target, sub_model)
    
    if metrics is not None:
        print("\n  [subgraph] metrics (over flattened adjacency):")
        print(f"    hamming_norm         = {metrics['hamming_norm']:.4f}")
        print(f"    adjacency_similarity = {metrics['adjacency_similarity']:.4f}")
        print(f"    edge_jaccard         = {metrics['edge_jaccard']:.4f}")
        print(f"    precision            = {metrics['precision']:.4f}")
        print(f"    recall               = {metrics['recall']:.4f}")
        print(f"    f1                   = {metrics['f1']:.4f}")
