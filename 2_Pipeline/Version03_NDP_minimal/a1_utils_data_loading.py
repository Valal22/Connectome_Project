"""
Data loading utilities for C. elegans connectome datasets.

Supports:
- Cook et al. 2019 format
- Varshney et al. 2011 format
"""

import numpy as np
import pandas as pd


def load_cook_connectome(file_path, sheet_name="hermaphrodite chemical"):
    """
    Load the Cook et al. 2019 C. elegans connectome.
    
    Parameters
    ----------
    file_path : str
        Path to Excel file
    sheet_name : str
        Sheet name containing the adjacency data
        
    Returns
    -------
    A : np.ndarray
        (N, N) binary adjacency matrix
    neurons : list
        List of N neuron names
    """
    sheet = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    # Row 2: column neuron labels (postsynaptic targets) starting at column 3
    header_row_idx = 2
    col_lab = {}
    for j, val in enumerate(sheet.iloc[header_row_idx]):
        if isinstance(val, str):
            col_lab[val] = j

    # Column 2: row neuron labels (presynaptic sources) starting at row 3
    row_lab = {}
    for i, val in enumerate(sheet.iloc[:, 2]):
        if isinstance(val, str) and i > header_row_idx:
            row_lab[val] = i

    # Use only neurons that appear as both pre- and post-synaptic
    neurons = sorted(row_lab.keys())
    N = len(neurons)

    A = np.zeros((N, N), dtype=np.int32)
    for i, pre in enumerate(neurons):
        ri = row_lab[pre]
        for j, post in enumerate(neurons):
            cj = col_lab[post]
            val = sheet.iat[ri, cj]
            if isinstance(val, (int, float)) and not pd.isna(val) and val != 0:
                A[i, j] = 1  # binarize

    return A, neurons


def load_varshney_connectome(file_path, sheet_name="Sheet1", weighted=False):
    """
    Load the Varshney et al. 2011 C. elegans connectome.
    
    Extracts ONLY chemical directed connections (Type = 'S' or 'Sp').
    Skips: R/Rp (reciprocal records), EJ (gap junctions), NMJ (neuromuscular).
    
    Type meanings:
        S   = Send (monadic chemical synapse: Neuron 1 -> Neuron 2)
        Sp  = Send polyadic (Neuron 1 -> Neuron 2, part of multi-target synapse)
        R   = Receive (reciprocal record, not a separate edge)
        Rp  = Receive polyadic (reciprocal record)
        EJ  = Electrical Junction (gap junction, excluded)
        NMJ = NeuroMuscular Junction (excluded)
    
    Parameters
    ----------
    file_path : str
        Path to Varshney_NeuronConnectFormatted.xlsx
    sheet_name : str
        Sheet name (default "Sheet1")
    weighted : bool
        If True, A[i,j] = sum of Nbr (multiplicity).
        If False, A[i,j] = 1 (binarized).
    
    Returns
    -------
    A : np.ndarray
        (N, N) adjacency matrix (directed)
    neurons : list
        List of N neuron names (sorted)
    """
    sheet = pd.read_excel(file_path, sheet_name=sheet_name)

    # Filter to chemical synapse SEND connections only (S and Sp)
    chem_send = sheet[sheet['Type'].isin(['S', 'Sp'])].copy()

    # Aggregate by neuron pair (sum Nbr for same Neuron 1 -> Neuron 2)
    edges = chem_send.groupby(['Neuron 1', 'Neuron 2'])['Nbr'].sum().reset_index()

    # Get all unique neurons (both pre and post)
    pre_set = set(edges['Neuron 1'].unique())
    post_set = set(edges['Neuron 2'].unique())
    all_neurons = pre_set.union(post_set)
    neurons = sorted(all_neurons)
    N = len(neurons)

    # Create mapping neuron to index
    neuron_to_idx = {n: i for i, n in enumerate(neurons)}

    # Build adjacency matrix
    A = np.zeros((N, N), dtype=np.int32)
    for _, row in edges.iterrows():
        i = neuron_to_idx[row['Neuron 1']]
        j = neuron_to_idx[row['Neuron 2']]
        if weighted:
            A[i, j] = int(row['Nbr'])
        else:
            A[i, j] = 1

    return A, neurons
