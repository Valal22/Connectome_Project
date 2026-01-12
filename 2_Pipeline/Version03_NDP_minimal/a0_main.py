import argparse
from pathlib import Path
import yaml
import networkx as nx
import numpy as np

from a1_utils_data_loading import load_cook_connectome, load_varshney_connectome
from a2_utils_eval_metrics import adjacency_to_digraph, compute_eval_metrics
from a3_ndp_generator import grow_network, count_ndp_parameters
from a4_train_metr import compute_train_metrics
from a5_cmaes_optimizer import run_cmaes

here = Path(__file__).resolve().parent
default_config = here / "config.yaml"
default_data = here / "SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx"


###############
# Helpers for YAML metric serialization
###############
def _serialize_metric_dict(d):
    """
    Convert metric dict values to YAML-safe Python scalars.
    Keeps non-numeric sentinel values (e.g. 'shape_mismatch') as strings.
    """
    out = {}
    for k, v in (d or {}).items():
        if v is None:
            out[k] = None
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v)
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            # Keep YAML-friendly containers as-is
            if isinstance(v, (list, dict)):
                out[k] = v
            else:
                # e.g. "shape_mismatch", "ok", error messages, etc.
                out[k] = str(v)

    return out

    
###############
# Lodaing configurations from yaml file
###############
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


###############
# CMA-ES Pipeline 
###############

def run_optimization(config):
    """
    Main optimization pipeline:
    1. Load target connectome
    2. Run CMA-ES
    3. Generate NDP graphs for gen1, gen5, best
    4. Compute all metrics
    5. Save results
    """
    
    # 1. Load target 
    file_path = config.get('file_path', str(default_data))
    sheet_name = config.get('sheet_name', 'hermaphrodite chemical')
    dataset_type = config.get('dataset_type', 'cook')

    if dataset_type == "varshney":
        A_target, neurons = load_varshney_connectome(file_path, sheet_name=sheet_name)
    else:
        A_target, neurons = load_cook_connectome(file_path, sheet_name=sheet_name)

    N = A_target.shape[0]
    target_edges = int(A_target.sum())
    target_density = target_edges / (N * N)

    print(f"\nTarget connectome: {N} nodes, {target_edges} edges, density={target_density:.4f}")

    # Computing the eval metrics for the target
    G_target = adjacency_to_digraph(A_target, neurons)
    target_eval = compute_eval_metrics(G_target)

    # Setting a penalty: if over a specific amount of edges (max_edges) a loss will be added. It's optional but it seems to work in producing graphs with a similar amount of edges as the target connectome
    if config.get('edge_penalty', False):
        config['max_edges'] = int(target_edges * config.get('max_edges_multiplier', 1.5))

    # Count NDP parameters
    nb_params = count_ndp_parameters(config)
    print(f"NDP trainable parameters: {nb_params}")

    # 2. Run CMA-ES optimization
    cmaes_result = run_cmaes(A_target, config)
    seed = config['seed']

    # 3. Generate NDP graphs for gen1, gen5, best
    def generate_ndp_data(params, label=""):
        """Generate graph from NDP parameters and compute adjacency."""
        if params is None:
            return None, None, None
        
        G, network_state = grow_network(params, config, seed=seed)
        A = nx.to_numpy_array(G, dtype=int)
        n_nodes = G.number_of_nodes()
        n_edges = int(A.sum())
        
        print(f"  {label}: {n_nodes} nodes, {n_edges} edges")
        return A, G, network_state

    print("\nGenerating graphs from optimized parameters:")
    A_ndp_gen1, G_ndp_gen1, _ = generate_ndp_data(cmaes_result['gen1_params'], "Gen1")
    A_ndp_gen5, G_ndp_gen5, _ = generate_ndp_data(cmaes_result['gen5_params'], "Gen5")
    A_ndp_best, G_ndp_best, _ = generate_ndp_data(cmaes_result['best_params'], "Best")

    # 4. Compute training metrics (only if shapes match)
    def safe_train_metrics(A_target, A_model, label=""):
        """Compute training metrics, handling shape mismatch."""
        if A_model is None:
            return {"status": "no_params"}
        
        if A_target.shape != A_model.shape:
            return {
                "status": "shape_mismatch",
                "target_shape": list(A_target.shape),
                "model_shape": list(A_model.shape)
            }
        
        metrics = compute_train_metrics(A_target, A_model)
        if metrics is None:
            return {"status": "compute_error"}
        
        metrics["status"] = "ok"
        return metrics

    train_ndp_gen1 = safe_train_metrics(A_target, A_ndp_gen1, "Gen1")
    train_ndp_gen5 = safe_train_metrics(A_target, A_ndp_gen5, "Gen5")
    train_ndp_best = safe_train_metrics(A_target, A_ndp_best, "Best")

    # 5. Compute eval metrics (only if shapes match and graph is reasonable)
    def safe_eval_metrics(G, label=""):
        """Compute eval metrics, handling errors."""
        if G is None:
            return {"status": "no_graph"}
        
        try:
            metrics = compute_eval_metrics(G, verbose=False)
            metrics["status"] = "ok"
            return metrics
        except Exception as e:
            return {"status": "compute_error", "error": str(e)}

    # Only compute eval for gen5 and best (gen1 often has wrong size)
    eval_ndp_gen5 = safe_eval_metrics(G_ndp_gen5, "Gen5") if G_ndp_gen5 else {"status": "no_graph"}
    eval_ndp_best = safe_eval_metrics(G_ndp_best, "Best") if G_ndp_best else {"status": "no_graph"}

    # 6. Build results structure
    Results = {
        'target': {
            'dataset_type': dataset_type,
            'nodes': N,
            'edges': target_edges,
            'density': float(target_density),
            'eval_metrics': _serialize_metric_dict(target_eval),
        },
        'optimization': {
            'method': 'NDP',
            'metric': config['metric'],
            'generations': cmaes_result['total_generations'],
            'popsize': config['popsize'],
            'seed': seed,
            'nb_parameters': nb_params,
            'growth_cycles': config['number_of_growth_cycles'],
        },
        'ndp_gen1': {
            'generation': 1,
            'nodes': int(G_ndp_gen1.number_of_nodes()) if G_ndp_gen1 else None,
            'edges': int(A_ndp_gen1.sum()) if A_ndp_gen1 is not None else None,
            'train_metrics': _serialize_metric_dict(train_ndp_gen1),
        },
        'ndp_gen5': {
            'generation': 5,
            'nodes': int(G_ndp_gen5.number_of_nodes()) if G_ndp_gen5 else None,
            'edges': int(A_ndp_gen5.sum()) if A_ndp_gen5 is not None else None,
            'train_metrics': _serialize_metric_dict(train_ndp_gen5),
            'eval_metrics': _serialize_metric_dict(eval_ndp_gen5),
        },
        'ndp_best': {
            'generation': cmaes_result['best_generation'],
            # 'score': float(cmaes_result['best_score']),
            'nodes': int(G_ndp_best.number_of_nodes()) if G_ndp_best else None,
            'edges': int(A_ndp_best.sum()) if A_ndp_best is not None else None,
            'train_metrics': _serialize_metric_dict(train_ndp_best),
            'eval_metrics': _serialize_metric_dict(eval_ndp_best),
        },
    }

    # Save results
    output_dir = Path(config.get('output_dir', 'results'))
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'Results_NDP.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(Results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # Also save best parameters
    if cmaes_result['best_params'] is not None:
        np.save(output_dir / 'best_params.npy', cmaes_result['best_params'])

    return Results, output_path

# CLI argument parsing
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize NDP parameters using CMA-ES")
    parser.add_argument("--config", type=str, default=str(default_config))
    parser.add_argument("--file-path", type=str)
    parser.add_argument("--sheet-name", type=str)
    parser.add_argument("--dataset-type", type=str, choices=["cook", "varshney"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    cli_overrides = {
        'file_path': args.file_path,
        'sheet_name': args.sheet_name,
        'dataset_type': args.dataset_type,
    }
    
    for key, val in cli_overrides.items():
        if val is not None:
            config[key] = val

    Results, output_path = run_optimization(config)
    print(f"\nResults saved to: {output_path}")


# Example usage:
# python a0_main.py
# python a0_main.py --dataset-type varshney --file-path Varshney_NeuronConnectFormatted.xlsx --sheet-name "Sheet1"