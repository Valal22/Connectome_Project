

# Pipeline exaplantion

This repository implements a simple baseline pipeline showing how datasets are loaded, how to compute evaluation and training metrics and a simple comparison between the target (connectome) graph and a generated one via Erdos-Renyi graph generator.

## Repository structure

- **`a0_main.py`**: Loads the connectome, computes target graph metrics, runs an Erdos–Renyi random graph generator and a “training” loop over multiple epochs, and prints global + subgraph metrics for each epoch. Implements a command line interface (CLI) with options for datasets path, number of epochs, edge probability, and random seed.  

- **`a1_utils_data_loading.py`**: (default) Contains `load_hermaphrodite_chemical_connectome`, which reads the Cook et al. connectome file, extracts neurons that appear as both pre- and post-synaptic, and returns a **binarized adjacency matrix** and the corresponding neuron labels.   

- **`a2_utils_eval_metrics.py`**: Provides `adjacency_to_digraph` to convert an adjacency matrix into a NetworkX `DiGraph`, and `compute_eval_metrics` to compute in-/out-degrees, average shortest path and hub neurons (others in future) on the largest strongly connected component. 

- **`a3_models_rand_generator.py`**: Implements `generate_er_random_graph`, which generates a directed Erdos–Renyi graph using NetworkX and returns both the adjacency matrix and the `DiGraph`.
    

- **`a4_models_train_metr.py`**:  
    - Defines `train_metrics`, which flattens adjacency matrices and computes Hamming distance, adjacency similarity, edge-level Jaccard, precision, recall, and F1 using scikit-learn. 
    - These metrics can be used both for evaluation and as a scalar fitness value and since we're using connectomes (which are sparse) maybe it's better to use edge F1 or edge Jaccard as optimisation objective. 
    - Also includes `print_subgraph` to display target vs model adjacency matrices for a small set of neurons and their local metrics.
    

---

## Dependencies
- Clone project: `git clone https://github.com/Valal22/Connectome_Project.git`

- Dependencies: `numpy`, `pandas`, `networkx`, `scikit-learn`

- Install via: `pip install numpy pandas networkx scikit-learn`

## How to run
1. **Default run**: Once downloaded > `cd Connectome_Project/2_Pipeline` > `python a0_main.py`

2. **Non-default runs (custom configuration)**. Other than changing file path, it's possible to override any of these CLI arguments:
    - **Change the number of epochs**:
        - `python a0_main.py --file-path SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx --num-epochs 10`
    - **Change ER edge probability p**:
        - `python a0_main.py --file-path SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx --num-epochs 5 --p 0.03`
    - **Change the random seed**:
        - `python a0_main.py --file-path SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx --num-epochs 5 --p 0.0413 --seed 42`
    - **Combine all of them.**


## Notes
1. a1_utils_data_loading:
    - To-do: Implementing other datasets types loading

2. To-do: Right implementation of (1) clustering, (2) small-world coefficient, (3) hubs, (4) motifs and replicate related results from Varshney et al. 2011, Towlson et al. 2013 and get the same metrics values 
    - Average shortest path seems to match the one of Varshney 

3. To-do: Switch from rand generator to NDP 
