# Neural Developmental Program for Connectome Growth


This repository trains and evaluates Neural Developmental Program (NDP) that grows a directed graph. The model starts from an initial graph, updates node embeddings through message passing, predicts node birth events, and wires edges using learned MLP rules. The target graph is built from C.elegans connectome data and evaluated with graph-level and edge-level metrics.

## Files
Guidelines for folder Version10_NDP_connectome

| File | Purpose |
|---|---|
| `a00_main.py` | Main entry point. Loads config/data, prepares embeddings, runs optimization, saves outputs. |
| `a10_data_load.py` | Loads and preprocesses connectome, neuron metadata, coordinates, birth order, cell types, and transcriptomics features. |
| `a30_ndp_generator.py` | Core NDP logic: graph initialization, message passing, node birth, edge wiring, and embedding handling. |
| `a40_train.py` | Training metrics. |
| `a50_optimizers.py` | CMA-ES optimization loop and fitness function. |
| `eval.py` | Evaluation metrics. |
| `config.yaml` | Main configuration file controlling data, embeddings, model, optimization, and evaluation settings. |
| `requirements.txt` | Python dependencies. |
| `python_version.txt` | Python version used for the environment. |

## Requirements

This project was run with: `Python 3.11.4`


## How to run
1. **Create and activate a virt env. On windows:** 
```
python -m venv .venv
.venv\Scripts\activate
```

2. **Install dependencies**
```
pip install -r requirements.txt
```

3. **Run training**
```
python a00_main.py --config config.yaml
```

4. **Evaluate**. After results are generated a new folder (/results) is generated with results. Enter /results folder, copy and paste the pkl and then rename it as bests_graph. Then cut and paste the renamed pkl file in the main folder (/Version10_NDP_connectome). After that run: `python eval.py`. 
