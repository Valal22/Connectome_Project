import argparse
from pathlib import Path
import yaml
import networkx as nx
import numpy as np

from a1_utils_data_loading import load_cook_connectome, load_varshney_connectome
from a2_utils_eval_metrics import adjacency_to_digraph, compute_eval_metrics
from a3_models_rand_generator import generate_er_graph, generate_sbm_graph
from a4_models_train_metr import compute_train_metrics, print_subgraph
from a5_cmaes_optimizer import run_cmaes

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml" 
DEFAULT_DATA = HERE / "SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx"


###############
# Lodaing configurations from yaml file
###############
def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)



###############
# CMA-ES Pipeline 
###############

def run_optimization(config):
    """
    Pipeline of the optimization:
    1. Load target connectome
    2. Run CMA-ES
    3. Generate SBM graphs for first generation (gen1), gen5, best
    4. Generate ER baseline
    5. Compute all metrics (for target connectome, SBM and ER)
    6. Save Results_txt in a yaml file
    """
    
    # 1. Load target
    file_path = config.get('file_path', str(DEFAULT_DATA))
    sheet_name = config.get('sheet_name', 'hermaphrodite chemical')
    dataset_type = config.get('dataset_type', 'cook')

    if dataset_type == "varshney":
        A_target, neurons = load_varshney_connectome(file_path, sheet_name=sheet_name)
    else:  # The cook dataset is the default one
        A_target, neurons = load_cook_connectome(file_path, sheet_name=sheet_name)

    N = A_target.shape[0]
    target_edges = int(A_target.sum())
    target_density = target_edges / (N * N)     # target_density is the fraction of possible edges and it's the ratio between the actual edges and the max possible ones given by N*N. The result is almost 3% > i.e. 3% of possible connections actually exist (sparse)

    # Computing the eval metrics for the target
    G_target = adjacency_to_digraph(A_target, neurons)
    target_eval = compute_eval_metrics(G_target)
    
    # Setting a penalty: if over a specific amount of edges (max_edges) a loss will be added. It's optional but it seems to work in producing graphs with a similar amount of edges as the target connectome
    if config.get('edge_penalty', False):
        config['max_edges'] = int(target_edges * config['max_edges_multiplier'])

    # 2. Run CMA-ES optimization
    cmaes_result = run_cmaes(A_target, config)
    seed = config['seed']

    # 3. Generate SBM for gen1, gen5, best
    def generate_sbm_data(params):
        G, _, _ = generate_sbm_graph(N, p_in=params[0], p_out=params[1], seed=seed)
        A = nx.to_numpy_array(G, dtype=int)
        return A, G

    A_sbm_gen1, G_sbm_gen1 = generate_sbm_data(cmaes_result['gen1_params'])
    A_sbm_gen5, G_sbm_gen5 = generate_sbm_data(cmaes_result['gen5_params'])
    A_sbm_best, G_sbm_best = generate_sbm_data(cmaes_result['best_params'])

    # 4. Generate ER baseline
    A_er, G_er = generate_er_graph(N, p=target_density, seed=seed)

    # 5. Compute train metrics for the 3 chosen SBM graph and the "baseline" ER graph
    train_sbm_gen1 = compute_train_metrics(A_target, A_sbm_gen1)
    train_sbm_gen5 = compute_train_metrics(A_target, A_sbm_gen5)
    train_sbm_best = compute_train_metrics(A_target, A_sbm_best)
    train_er = compute_train_metrics(A_target, A_er)

    # 5. Compute eval metrics (only for gen5, best, ER - gen1 has bad params: since I set random initial probabilities theyr number is high > this means it generated a lot of edges and computing eval metrics over that graph would take too much time)
    eval_sbm_gen5 = compute_eval_metrics(G_sbm_gen5)
    eval_sbm_best = compute_eval_metrics(G_sbm_best)
    eval_er = compute_eval_metrics(G_er)

    # 6. Results file structure generation
    Results_txt = {
        'target': {
            'dataset_type': dataset_type,
            'nodes': N,
            'edges': target_edges,
            'density': float(target_density),
            'eval_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in target_eval.items()},
        },
        'optimization': {
            'metric': config['metric'],
            'generations': cmaes_result['total_generations'],
            'popsize': config['popsize'],
            'seed': seed,
        },
        'sbm_gen1': {
            'generation': 1,
            'params': {'p_in': float(cmaes_result['gen1_params'][0]), 
                      'p_out': float(cmaes_result['gen1_params'][1])},
            'solutions': [[float(s[0]), float(s[1])] for s in cmaes_result['gen1_cand_sol']],
            'nodes': int(G_sbm_gen1.number_of_nodes()),
            'edges': int(A_sbm_gen1.sum()),
            'train_metrics': {k: float(v) for k, v in train_sbm_gen1.items()},
        },
        'sbm_gen5': {
            'generation': 5,
            'params': {'p_in': float(cmaes_result['gen5_params'][0]), 
                      'p_out': float(cmaes_result['gen5_params'][1])},
            'solutions': [[float(s[0]), float(s[1])] for s in cmaes_result['gen5_cand_sol']],
            'nodes': int(G_sbm_gen5.number_of_nodes()),
            'edges': int(A_sbm_gen5.sum()),
            'train_metrics': {k: float(v) for k, v in train_sbm_gen5.items()},
            'eval_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in eval_sbm_gen5.items()},
        },
        'sbm_best': {
            'generation': cmaes_result['best_generation'],
            'params': {'p_in': float(cmaes_result['best_params'][0]), 
                      'p_out': float(cmaes_result['best_params'][1])},
            'score': float(cmaes_result['best_score']),
            'nodes': int(G_sbm_best.number_of_nodes()),
            'edges': int(A_sbm_best.sum()),
            'train_metrics': {k: float(v) for k, v in train_sbm_best.items()},
            'eval_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in eval_sbm_best.items()},
        },
        'er': {
            'p': float(target_density),
            'nodes': int(G_er.number_of_nodes()),
            'edges': int(A_er.sum()),
            'train_metrics': {k: float(v) for k, v in train_er.items()},
            'eval_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else int(v) 
                           for k, v in eval_er.items()},
        },
        # 'history': cmaes_result['history'],
    }

    # Save Results_txt as a yaml file
    output_dir = Path(config.get('output_dir', 'Results_txt'))
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'Results_txt.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(Results_txt, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return Results_txt, output_path

# CLI argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Optimize SBM parameters using CMA-ES")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--sheet-name", type=str)
    parser.add_argument("--dataset-type", type=str, choices=["cook", "varshney"])
    parser.add_argument("--generations", type=int)
    parser.add_argument("--popsize", type=int)
    parser.add_argument("--sigma", type=float)
    parser.add_argument("--metric", type=str, 
                        choices=['f1', 'edge_jaccard', 'precision', 'recall', 'adjacency_similarity', 'hamming_norm'])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    for key in ['file_path', 'sheet_name', 'dataset_type', 'generations', 'popsize', 'sigma', 'metric', 'seed']:
        val = getattr(args, key.replace('-', '_'), None)
        if val is not None:
            config[key] = val

    Results_txt, output_path = run_optimization(config)
    print(f"Results saved to: {output_path}")



# Example usage:
# python a0_main.py
# python a0_main.py --dataset-type varshney --file-path Varshney_NeuronConnectFormatted.xlsx --sheet-name "Sheet1"