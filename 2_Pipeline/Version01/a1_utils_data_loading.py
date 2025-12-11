import numpy as np
import pandas as pd


############################
# 1. Data loading - Cook
############################

def load_cook_connectome(file_path, sheet_name="hermaphrodite chemical"):
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
                A[i, j] = 1  # binarize step


    return A, neurons
        # A: (N, N) adjacency matrix
        # neurons: list of N neuron names


############################
# 2. Data loading - Varshney
############################
def load_varshney_connectome(file_path, sheet_name="Sheet1", weighted=False):
    sheet = pd.read_excel(file_path, sheet_name=sheet_name)
    

    # Filter to chemical synapse SEND connections only (S and Sp). S  should be monadic, Sp should be polyadic.
    # Skip R, Rp (reciprocal/receiving connections), EJ (gap junctions), NMJ (neuromuscular)
    # Note: for understanding where R, Rp, Neuron 1, Nbr, etc come from look at the dataset  
    chem_send = sheet[sheet['Type'].isin(['S', 'Sp'])].copy()
    

    # Aggregate by neuron pair (sum Nbr for same Neuron 1 -> Neuron 2), e.g.: just count as 1 connection between 2 neurons even if multiple connections with different weights (so there will be just 1 row per unique directed edge) 
    edges = chem_send.groupby(['Neuron 1', 'Neuron 2'])['Nbr'].sum().reset_index()
    

    # Get all unique neurons (both pre and post)
    pre_set = set(edges['Neuron 1'].unique())   # all presynaptic neurons
    post_set = set(edges['Neuron 2'].unique())  # all postsynaptic neurons
    all_neurons = pre_set.union(post_set)   
    neurons = sorted(all_neurons)   
    N = len(neurons)
    

    # Create mapping neuron to index 
    neuron_to_idx = {n: i for i, n in enumerate(neurons)}
    

    # Building the adjacency matrix and binarizing the connections 
    A = np.zeros((N, N), dtype=np.int32)
    for _, row in edges.iterrows():             # row: Neuron 1, Neuron 2, Nbr; row=each edge (see edges)
        i = neuron_to_idx[row['Neuron 1']]      # i = each presynaptic
        j = neuron_to_idx[row['Neuron 2']]      # j = each postsynaptic
        if weighted:
            A[i, j] = int(row['Nbr'])
        else:
            A[i, j] = 1  # same binarize step
    

    return A, neurons

