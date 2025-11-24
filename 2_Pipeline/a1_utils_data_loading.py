import numpy as np
import pandas as pd


############################
# 1. Data loading utilities
############################

def load_hermaphrodite_chemical_connectome(file_path, sheet_name="hermaphrodite chemical",):
    sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Row 2: column neuron labels (postsynaptic targets) starting at column 3
    header_row_idx = 2
    col_labels = {}
    for j, val in enumerate(sheet.iloc[header_row_idx]):
        if isinstance(val, str):
            col_labels[val] = j

    # Column 2: row neuron labels (presynaptic sources) starting at row 3
    row_labels = {}
    for i, val in enumerate(sheet.iloc[:, 2]):
        if isinstance(val, str) and i > header_row_idx:
            row_labels[val] = i

    # Use only neurons that appear as both pre- and post-synaptic
    neurons = sorted(row_labels.keys())
    N = len(neurons)

    A = np.zeros((N, N), dtype=np.int32)
    for i, pre in enumerate(neurons):
        ri = row_labels[pre]
        for j, post in enumerate(neurons):
            cj = col_labels[post]
            val = sheet.iat[ri, cj]
            if isinstance(val, (int, float)) and not pd.isna(val) and val != 0:
                A[i, j] = 1  # binarize

    return A, neurons
        # A: (N, N) adjacency matrix
        # neurons: list of N neuron names