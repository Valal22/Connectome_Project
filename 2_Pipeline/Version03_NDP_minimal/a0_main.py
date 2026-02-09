import argparse
from pathlib import Path
import yaml
import networkx as nx
import numpy as np
import pandas as pd
import wandb

import glob
import imageio.v2 as imageio

import os



from a1_utils_data_loading import (
    load_cook_connectome, 
    load_varshney_connectome, 
    load_neuron_birth_order, 
)
from a2_utils_eval_metrics import adjacency_to_digraph, compute_eval_metrics
from a3_ndp_generator import grow_network, count_ndp_parameters
from a4_train_metr import compute_train_metrics
from a5_cmaes_optimizer import run_cmaes

here = Path(__file__).resolve().parent
default_config = here / "config.yaml"
default_data = here / "SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx"


def frames_to_video(frames_dir, out_path, fps=10):
    frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frames:
        print(f"[video] No frames found in {frames_dir}")
        return

    # Try MP4 first, fallback to GIF if needed
    ext = os.path.splitext(out_path)[1].lower()
    if ext == ".mp4":
        try:
            # with imageio.get_writer(out_path, fps=fps) as w:
            with imageio.get_writer(
                out_path, 
                fps=fps, 
                macro_block_size=1,
                ffmpeg_log_level='error',  # Suppress info/warning messages
                input_params=['-r', str(fps)],  # Explicit input frame rate
            ) as w:
                for f in frames:
                    img = imageio.imread(f)
                    """
                    w.append_data(img)
                    w.append_data(img)   # 2x slower
                    w.append_data(img)   # 3x slower
                    w.append_data(img)   # 4x slower
                    w.append_data(img)   # 5x slower
                    w.append_data(img)   # 6x slower
                    """

                    # Or:
                    slow_factor = 12  # 8x slower
                    for _ in range(slow_factor):
                        w.append_data(img) 
            
            print(f"[video] Saved {out_path}")
            return
        except Exception as e:
            print(f"[video] MP4 failed ({e}). Falling back to GIF.")
            out_path = out_path[:-4] + ".gif"

    with imageio.get_writer(out_path, mode="I", duration=(2.0/fps)) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    print(f"[video] Saved {out_path}")

# ====================================
# Helpers for YAML metric serialization
# ====================================
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
            if isinstance(v, (list, dict)):
                out[k] = v
            else:
                out[k] = str(v)
    return out

# ====================================
# Lodaing configurations from yaml file
# ====================================
def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)

# ====================================
# CMA-ES Pipeline 
# ====================================
def run_optimization(config):
    """
    Main optimization pipeline:
    1. Load target connectome
    2. Run CMA-ES
    3. Generate NDP graphs for gen1, gen5, best
    4. Compute all metrics
    5. Save results
    """

    use_wandb = bool(config["use_wandb"])
    wandb_run = None

    if use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_run_name"],
            config=config,   # logs your full YAML config
        )

        # Log target summary early
        wandb.summary["dataset_type"] = config["dataset_type"]

    # 1. Load target 
    file_path = config['file_path']
    sheet_name = config['sheet_name']
    dataset_type = config['dataset_type']

    if dataset_type == "varshney":
        birth_csv_path = config["birth_csv_path"]
        birth_order = load_neuron_birth_order(birth_csv_path)
        allowed_neurons = set(birth_order)

        A_target, neurons = load_varshney_connectome(
            file_path,
            sheet_name=sheet_name,
            weighted=False,
            allowed_neurons=allowed_neurons,
            neuron_order=birth_order,
        )

        if config["target_n_order"]:
            print(f"All neurons ordered are: {neurons}")

        config["neurons"] = neurons

    else:
        A_target, neurons = load_cook_connectome(file_path, sheet_name=sheet_name)

    # Optional: train only on first K neurons
    K = config["eval_first_n"]
    if K is not None:
        K = int(K)
        A_target = A_target[:K, :K]
        neurons = neurons[:K]
        config["neurons"] = neurons
        if config["target_n_order"]:
            print(f"Evaluated neurons ordered are: {neurons}")


        config["target_network_size"] = K

    N = A_target.shape[0]
    target_edges = int(A_target.sum())
    target_density = target_edges / (N * N)

    print(f"\nTarget connectome: {N} nodes, {target_edges} edges, density={target_density:.4f}")

    # Analyze target degree distribution
    in_deg = A_target.sum(axis=0)
    out_deg = A_target.sum(axis=1)
    print(f"  Out-degree: min={out_deg.min()}, max={out_deg.max()}, mean={out_deg.mean():.2f}")
    print(f"  In-degree:  min={in_deg.min()}, max={in_deg.max()}, mean={in_deg.mean():.2f}")

    # Computing the eval metrics for the target
    G_target = adjacency_to_digraph(A_target, neurons)
    target_eval = compute_eval_metrics(G_target)

    # Count NDP parameters
    nb_params = count_ndp_parameters(config)
    if config["count_ndp_parameters_prints"]:
        print(f"\nNDP parameters breakdown:")
        print(f"  Embedding: {config['nb_params_coevolve_initial_embeddings']}")
        print(f"  Growth MLP: {config['nb_params_growth_model']}")
        print(f"  Transform MLP: {config['nb_params_feature_transformation']}")
        print(f"  Edge MLP: {config['nb_params_edge_model']}")
        print(f"  TOTAL: {nb_params}")

    # 2. Run CMA-ES optimization
    cmaes_result = run_cmaes(A_target, config)
    seed = config['seed']
    
    # 3. Generate NDP graphs for gen1, gen5, best
    def generate_ndp_data(params, label=""):
        """Generate graph from NDP parameters and compute adjacency."""
        if params is None:
            return None, None, None

        # Phase-aware reconstruction 
        G, network_state = grow_network(params, config, seed=config["seed"], verbose=False, debug=False)
        
        A = nx.to_numpy_array(G, dtype=int)

        n_nodes = G.number_of_nodes()
        n_edges = int(A.sum())
        print(f"  {label}: {n_nodes} nodes, {n_edges} edges")
        return A, G, network_state


    print("\nGenerating graphs from optimized parameters:")
    A_ndp_gen1, G_ndp_gen1, _ = generate_ndp_data(cmaes_result['gen1_params'], "Gen1")
    A_ndp_gen5, G_ndp_gen5, _ = generate_ndp_data(cmaes_result['gen5_params'], "Gen5")
    A_ndp_best, G_ndp_best, _ = generate_ndp_data(cmaes_result['best_params'], "Best")

    
    # 4. Compute training metrics
    def safe_train_metrics(A_target, A_model, label=""):
        # Compute training metrics, handling shape mismatch.
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
    
        # Note: this function is notifying is shapes mismatch BUT what it actually works
        # is to consider also when they do not match (implemented in fitness_function in a5 file)
        # This function should be fixed/removed  

    # 5. Compute eval metrics (only if shapes match and graph is reasonable)
    train_ndp_gen1 = safe_train_metrics(A_target, A_ndp_gen1, "Gen1")
    train_ndp_gen5 = safe_train_metrics(A_target, A_ndp_gen5, "Gen5")
    train_ndp_best = safe_train_metrics(A_target, A_ndp_best, "Best")

    # Compute eval metrics
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
            'phase3_edge_mode': 'global_topE',
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
            'nodes': int(G_ndp_best.number_of_nodes()) if G_ndp_best else None,
            'edges': int(A_ndp_best.sum()) if A_ndp_best is not None else None,
            'train_metrics': _serialize_metric_dict(train_ndp_best),
            'eval_metrics': _serialize_metric_dict(eval_ndp_best),
        },
    }

    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)


    # ====================================
    # VIDEO: growth of BEST graph (node-by-node, edge-by-edge)
    # ====================================
    frames_dir = os.path.join(output_dir, "growth_frames_best")
    
    os.makedirs(frames_dir, exist_ok=True)

    
    G_best, state_best = grow_network(
        cmaes_result["best_params"],
        config,
        seed=config["seed"],
        record_video=True,
        video_frames_dir=frames_dir
    )

    video_path = os.path.join(output_dir, "growth_best.mp4")
    frames_to_video(frames_dir, video_path, fps=12)    

    # ====================================
    # Log growth video to Weights & Biases
    # ====================================
    if use_wandb and os.path.exists(video_path):
        wandb.log({
            "growth_video_best": wandb.Video(video_path, fps=12, format="mp4")
        })

        # Optional: log frames directory as artifact
        growth_artifact = wandb.Artifact("growth_frames_best", type="growth-frames")
        for f in glob.glob(os.path.join(frames_dir, "frame_*.png")):
            growth_artifact.add_file(f)

        wandb.log_artifact(growth_artifact)


    output_path = output_dir / 'Results_NDP.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(Results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if cmaes_result['best_params'] is not None:
        np.save(output_dir / 'best_params.npy', cmaes_result['best_params'])

    return Results, output_path

# CLI argument parsing
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize NDP parameters using CMA-ES (FIXED V2)")
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

    if wandb.run is not None:
        import wandb

        art = wandb.Artifact("ndp_outputs", type="results")
        art.add_file(str(output_path))  # Results_NDP.yaml

        output_dir = Path(output_path).parent
        best_params_path = output_dir / "best_params.npy"
        if best_params_path.exists():
            art.add_file(str(best_params_path))

        if Results.get("ndp_phase3_best_adj_xlsx"):
            art.add_file(Results["ndp_phase3_best_adj_xlsx"])

        wandb.log_artifact(art)

        # Nice-to-have: store final scores in W&B summary
        wandb.summary["final/best_gen"] = Results["ndp_best"]["generation"]

        best_tm = Results["ndp_best"]["train_metrics"]
        if best_tm.get("status") != "ok":
            raise RuntimeError(f"ndp_best train_metrics not ok: {best_tm}")

        wandb.summary["final/f1"] = best_tm["f1"]
        wandb.summary["final/jaccard"] = best_tm["edge_jaccard"]

# Example usage:
# python a0_main.py
# python a0_main.py --config config.yaml --dataset-type varshney --file-path Varshney_NeuronConnectFormatted.xlsx --sheet-name Sheet1
