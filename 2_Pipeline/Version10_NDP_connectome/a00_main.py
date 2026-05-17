from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import networkx as nx
import pickle 
import argparse
import wandb

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime


from a10_data_load import load_witv, adj_in_birth_t, from_csv, to_n_emb, load_neuron_xyz, processed_coords, to_n_emb_raw, load_gene_features
from a20_stats_eval import adjacency_to_digraph, compute_eval_metrics 
from a30_ndp_generator import grow_network, count_ndp_parameters, embeddings_dimensions
from a40_train import compute_train_metrics
from a50_optimizers import run_cmaes 

HERE = Path.cwd()  
DATASETS = HERE / "datasets"
CONFIG = HERE / "config.yaml"
ADJ_MATRIX = DATASETS / "synapse_count_matrices.xlsx"
COORDS_PATH = DATASETS / "NeuronPosition3D.csv"
TRANSC_PATH = DATASETS / "transcript.csv"




def load_config(CONFIG):
    with open(CONFIG, 'r') as file:
        config = yaml.safe_load(file)
    return config


# ===============================================
# Convert compute_eval_metrics tuple to yaml dict
# ===============================================
def _eval_tuple_to_dict(t):
    nodes, edges, degrees_stats, avg_c, avg_pl, sw, n_hubs, recipr = t
    return {
        "nodes": int(nodes),
        "edges": int(edges),
        "in_degree_mean":     float(degrees_stats["in_degree_mean"]),
        "in_degree_std":      float(degrees_stats["in_degree_std"]),
        "out_degree_mean":    float(degrees_stats["out_degree_mean"]),
        "out_degree_std":     float(degrees_stats["out_degree_std"]),
        "total_degree_mean":  float(degrees_stats["total_degree_mean"]),
        "total_degree_std":   float(degrees_stats["total_degree_std"]),
        "avg_clustering":     float(avg_c),
        "avg_path_length":    float(avg_pl),
        "small_world_coeff":  float(sw),
        "n_hubs":             int(n_hubs),
        "reciprocity":        float(recipr),
    }



# ===============================================
# Preprocess fixed embeddings and stores them
# ===============================================
def prepare_static_node_features_once(config):
    common_names = config["common_names"]
    n_emb = config["n_emb"]

    mode, fixed_D, dynamic_D, mode_D, mode_D_fixed = embeddings_dimensions(config)

    if "_static_node_features_prepared" in config:
        if config["_static_node_features_prepared"] is True:
            raise RuntimeError("Static node features were already prepared. This function must be called exactly once.")
        raise ValueError("config['_static_node_features_prepared'] exists but is not True.")

    if mode_D_fixed == 0:
        all_emb = np.stack([
            n_emb[name].reshape(-1)
            for name in common_names
        ]).astype(np.float64)

        if all_emb.shape[1] != mode_D:
            raise ValueError(f"all_emb last dim {all_emb.shape[1]} != mode_D {mode_D}")

        emb_mean = all_emb.mean(axis=0, keepdims=True)
        emb_std = all_emb.std(axis=0, keepdims=True)

        if np.any(emb_std == 0.0):
            zero_std_columns = np.where(emb_std[0] == 0.0)[0].tolist()
            raise ValueError(f"Zero standard deviation in class embedding columns: {zero_std_columns}")

        n_emb_scaled = {}
        for name in common_names:
            row = n_emb[name].reshape(1, -1).astype(np.float64)
            n_emb_scaled[name] = (row - emb_mean) / emb_std

        config["n_emb"] = n_emb_scaled
        config["_proto_mean"] = emb_mean
        config["_proto_std"] = emb_std
        config["_static_node_features_prepared"] = True
        return

    if mode_D_fixed > 0 and config["process_altogether"]:
        class_stack = np.stack([
            n_emb[name].reshape(-1)
            for name in common_names
        ]).astype(np.float64)

        fixed_stack = np.stack([
            config["processed_fixed_feats"][name].reshape(-1)
            for name in common_names
        ]).astype(np.float64)

        if class_stack.shape[1] != mode_D:
            raise ValueError(f"class_stack last dim {class_stack.shape[1]} != mode_D {mode_D}")

        if fixed_stack.shape[1] != mode_D_fixed:
            raise ValueError(f"fixed_stack last dim {fixed_stack.shape[1]} != mode_D_fixed {mode_D_fixed}")

        all_emb = np.concatenate([class_stack, fixed_stack], axis=1)

        all_emb_norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
        if np.any(all_emb_norms == 0.0):
            zero_norm_rows = np.where(all_emb_norms[:, 0] == 0.0)[0].tolist()
            zero_norm_names = [common_names[i] for i in zero_norm_rows]
            raise ValueError(f"Zero-norm rows in combined embedding table: {zero_norm_names}")

        all_emb = all_emb / all_emb_norms

        class_stack = all_emb[:, :mode_D]
        fixed_stack = all_emb[:, mode_D:]

        if config["birth_noise"]:
            birth_range = float(config["node_emb_birth_uniform_range"])
            if birth_range < 0:
                raise ValueError(f"node_emb_birth_uniform_range must be >= 0, got {birth_range}")

            rng = np.random.default_rng(config["seed"])
            noise = rng.uniform(
                low=-birth_range,
                high=birth_range,
                size=class_stack.shape,
            ).astype(np.float64)

            class_stack = class_stack + noise

        all_emb = np.concatenate([class_stack, fixed_stack], axis=1)

        emb_mean = all_emb.mean(axis=0, keepdims=True)
        emb_std = all_emb.std(axis=0, keepdims=True)

        if np.any(emb_std == 0.0):
            zero_std_columns = np.where(emb_std[0] == 0.0)[0].tolist()
            raise ValueError(f"Zero standard deviation in combined embedding columns: {zero_std_columns}")

        all_emb_scaled = (all_emb - emb_mean) / emb_std

        class_scaled = all_emb_scaled[:, :mode_D]
        fixed_scaled = all_emb_scaled[:, mode_D:]

        n_emb_scaled = {}
        processed_fixed_feats_scaled = {}

        for i, name in enumerate(common_names):
            n_emb_scaled[name] = class_scaled[i].reshape(1, -1)
            processed_fixed_feats_scaled[name] = fixed_scaled[i].reshape(1, -1)

        config["n_emb"] = n_emb_scaled
        config["processed_fixed_feats"] = processed_fixed_feats_scaled
        config["_proto_mean"] = emb_mean
        config["_proto_std"] = emb_std
        config["_static_node_features_prepared"] = True
        return

    if mode_D_fixed > 0 and not config["process_altogether"]:
        all_emb = np.stack([
            n_emb[name].reshape(-1)
            for name in common_names
        ]).astype(np.float64)

        if all_emb.shape[1] != mode_D:
            raise ValueError(f"all_emb last dim {all_emb.shape[1]} != mode_D {mode_D}")

        emb_mean = all_emb.mean(axis=0, keepdims=True)
        emb_std = all_emb.std(axis=0, keepdims=True)

        if np.any(emb_std == 0.0):
            zero_std_columns = np.where(emb_std[0] == 0.0)[0].tolist()
            raise ValueError(f"Zero standard deviation in class embedding columns: {zero_std_columns}")

        n_emb_scaled = {}
        for name in common_names:
            row = n_emb[name].reshape(1, -1).astype(np.float64)
            n_emb_scaled[name] = (row - emb_mean) / emb_std

        config["n_emb"] = n_emb_scaled
        config["_proto_mean"] = emb_mean
        config["_proto_std"] = emb_std
        config["_static_node_features_prepared"] = True
        return

    raise ValueError(
        f"Unsupported static feature preparation case: "
        f"mode_D_fixed={mode_D_fixed}, process_altogether={config['process_altogether']}"
    )



def main_run(config):
    seed = config['seed']

    use_wandb = config['use_wandb']
    if use_wandb:
        import wandb
        wandb.init(
            project=config["wandb_project"],
            entity=config["wandb_entity"],
            name=config["wandb_run_name"],
            config=config,   
        )

        # Log target summary early
        wandb.summary["dataset_type"] = config["dataset_type"]

    # FILE_PATH = DATASETS / "synapse_count_matrices.xlsx"
    FILE_PATH = DATASETS / config["file_path"]
    sheet_name = config["sheet_name"]
    dataset_type = config["dataset_type"]
    
    if dataset_type == "witv":
        common_names, A_target = load_witv(file_path=ADJ_MATRIX, sheet_name=sheet_name)        
        neurons_list, n_features_dict, class_names_list = from_csv(FILE_PATH)
        common_names, A_target = adj_in_birth_t(common_names, A_target, neurons_list)


        if config["coords_emb"]:
            coords_dict = load_neuron_xyz(coords_path=COORDS_PATH, common_names=common_names)
            processed_coords_dict = processed_coords(coords_dict, config)
        else:
            processed_coords_dict = {}

        if config["transcr_emb"]:
            transc_genes = config["transc_genes"]
            config["genes"] = len(transc_genes)
            transc_features_dict = load_gene_features(
                transc_path=TRANSC_PATH,
                genes=transc_genes,
                neurons=common_names,
            )
        else:
            config["genes"] = 0
            transc_features_dict = {}

        if config["use_fixed_feats"] != (config["coords_emb"] or config["transcr_emb"]):
            raise ValueError(
                "config['use_fixed_feats'] must be True exactly when at least one fixed feature "
                "source is enabled: config['coords_emb'] or config['transcr_emb']."
            )

        mode, fixed_D, dynamic_D, mode_D, mode_D_fixed = embeddings_dimensions(config)

        config["common_names"] = list(common_names) 
        config["df_neurons"] = neurons_list
        config["n_features"] = n_features_dict
        config["class_names"] = class_names_list

        if mode_D_fixed ==0:
            config["n_emb"] = to_n_emb(
                class_names_list=class_names_list,
                config=config,
                neurons_list=neurons_list,
                n_features_dict=n_features_dict,
            )
        
        elif mode_D_fixed > 0 and config["process_altogether"]:
            config["n_emb"] = to_n_emb_raw(
                class_names_list=class_names_list,
                config=config,
                neurons_list=neurons_list,
                n_features_dict=n_features_dict,
            )

        elif mode_D_fixed > 0 and not config["process_altogether"]:
            config["n_emb"] = to_n_emb(
                class_names_list=class_names_list,
                config=config,
                neurons_list=neurons_list,
                n_features_dict=n_features_dict,
            )

        else:
            raise ValueError(f"Unsupported mode_D_fixed={mode_D_fixed}")

        neurons = list(common_names)


        if config["use_fixed_feats"]:
            if mode_D_fixed <= 0:
                raise ValueError(
                    f"config['use_fixed_feats'] is True but mode_D_fixed={mode_D_fixed}. "
                    "Enable config['coords_emb'] or config['transcr_emb']."
                )

            processed_fixed_dict = {}

            for neuron in common_names:
                fixed_parts = []

                if config["coords_emb"]:
                    fixed_parts.append(processed_coords_dict[neuron].reshape(-1))

                if config["transcr_emb"]:
                    fixed_parts.append(transc_features_dict[neuron].reshape(-1))

                if len(fixed_parts) == 0:
                    raise ValueError(
                        "config['use_fixed_feats'] is True but no fixed feature parts were assembled."
                    )

                processed_fixed_dict[neuron] = np.concatenate(fixed_parts).astype(np.float64)

            config["processed_fixed_feats"] = processed_fixed_dict
        else:
            if mode_D_fixed != 0:
                raise ValueError(
                    f"config['use_fixed_feats'] is False but mode_D_fixed={mode_D_fixed}. "
                    "Disable config['coords_emb'] and config['transcr_emb'], or set use_fixed_feats=True."
                )

        prepare_static_node_features_once(config)

        if config["pair_distance_feat"]:
            if not config["coords_emb"]:
                raise ValueError("pair_distance_feat=True requires coords_emb=True")

            common_names = config["common_names"]

            coords = np.stack([
                config["processed_fixed_feats"][name].reshape(-1)[:3]
                for name in common_names
            ]).astype(np.float64)

            diff = coords[:, None, :] - coords[None, :, :]
            dist = np.linalg.norm(diff, axis=2)

            mask = ~np.eye(dist.shape[0], dtype=bool)
            dist_values = dist[mask]

            dist_mean = float(dist_values.mean())
            dist_std = float(dist_values.std())

            if dist_std == 0.0:
                raise ValueError("Zero standard deviation in pairwise distance feature")

            config["_pair_dist_mean"] = dist_mean
            config["_pair_dist_std"] = dist_std


    # elif dataset_type == "varshney":

    # elif dataset_type == "cook":

    k = config["eval_first_n"]
    if k is not None:
        k = int(k) 

        A_target = A_target[:k, :k]


    N = A_target.shape[0] 
    target_edges = int(A_target.sum())
    target_density = target_edges / (N * N)

    config["target_edges"] = target_edges
    config["target_density"] = float(target_density)

    print(f"\nTarget connectome: {N} nodes, {target_edges} edges, density={target_density:.4f}")

    # Compute eval metrics for target
    G_target = adjacency_to_digraph(A_target, neurons)

    if config["compute_eval_metrics"]:
        target_eval = _eval_tuple_to_dict(compute_eval_metrics(G_target))
    else:
        target_eval = None 

    # Count NDP parameters
    nb_params = count_ndp_parameters(config)
    if config["count_ndp_parameters_prints"]:
        print(f"\nNDP parameters breakdown:")
        print(f"  Embedding: {config['nb_params_coevolve_initial_embeddings']}")
        print(f"  Growth MLP: {config['nb_params_growth_model']}")
        print(f"  Transform MLP: {config['nb_params_feature_transformation']}")
        print(f"  Edge MLP: {config['nb_params_edge_model']}")
        print(f"  TOTAL: {nb_params}")

    # Run CMA-ES optimization
    cmaes_result = run_cmaes(A_target, config)
    
    def generate_ndp_data(params, label="", verbose_growth=False):
        if params is None:
            return None, None, None, None 

        # Reconstruction 
        G, edge_network_state, growth_network_state = grow_network(params, config, seed=config["seed"], verbose=verbose_growth)
        
        A = nx.to_numpy_array(G, dtype=int)
        if "_soft_P" not in G.graph:
            raise KeyError("G.graph['_soft_P'] missing; set store_soft_P=True in config")
        soft_A = np.asarray(G.graph["_soft_P"], dtype=float)


        n_nodes = G.number_of_nodes()
        n_edges = int(A.sum())
        print(f"  {label}: {n_nodes} nodes, {n_edges} edges")
        return A, soft_A, G, (edge_network_state, growth_network_state)

    print("\nGenerating graphs from optimized parameters:")

    A_ndp_gen1, soft_ndp_gen1, G_ndp_gen1, _ = generate_ndp_data(
        cmaes_result["gen1_params"],
        "Gen1",
        verbose_growth=False,
    )

    A_ndp_gen5, soft_ndp_gen5, G_ndp_gen5, _ = generate_ndp_data(
        cmaes_result["gen5_params"],
        "Gen5",
        verbose_growth=False,
    )

    print("\nDebug trace for BEST candidate:")
    A_ndp_best, soft_ndp_best, G_ndp_best, _ = generate_ndp_data(
        cmaes_result["best_params"],
        f"Best gen {cmaes_result['best_generation']}",
        verbose_growth=True,
    )

    train_ndp_gen1 = compute_train_metrics(A_target, A_ndp_gen1, soft_ndp_gen1, config)
    train_ndp_gen5 = compute_train_metrics(A_target, A_ndp_gen5, soft_ndp_gen5, config)
    train_ndp_best = compute_train_metrics(A_target, A_ndp_best, soft_ndp_best, config)


    if config["compute_eval_metrics"]:
        if G_ndp_best is not None:
            eval_ndp_best = _eval_tuple_to_dict(compute_eval_metrics(G_ndp_best))
        else:
            eval_ndp_best = {"status": "no_graph"}
    else:
        eval_ndp_best = {"status": "skipped"}

    # Build results
    Results = {
        'target': {
            'dataset_type': dataset_type,
            'nodes': N,
            'edges': target_edges,
            'density': float(target_density),
            'eval_metrics': target_eval,
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
            'train_metrics': train_ndp_gen1,
        },
        'ndp_gen5': {
            'generation': 5,
            'nodes': int(G_ndp_gen5.number_of_nodes()) if G_ndp_gen5 else None,
            'edges': int(A_ndp_gen5.sum()) if A_ndp_gen5 is not None else None,
            'train_metrics': train_ndp_gen5,
        },
        'ndp_best': {
            'generation': cmaes_result['best_generation'],
            'nodes': int(G_ndp_best.number_of_nodes()) if G_ndp_best else None,
            'edges': int(A_ndp_best.sum()) if A_ndp_best is not None else None,
            'train_metrics': train_ndp_best,
            'eval_metrics': eval_ndp_best,
        },
    }

    # Save results
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(exist_ok=True)

    # Save best graph as pickle
    if G_ndp_best is not None:
        gen_tag = f"gen{cmaes_result['best_generation']}"
        best_graph_pkl_path = output_dir / f"best_graph_{gen_tag}.pkl"
        with open(best_graph_pkl_path, "wb") as f:
            pickle.dump(G_ndp_best, f, protocol=pickle.HIGHEST_PROTOCOL)
        Results["best_graph_pkl"] = str(best_graph_pkl_path)

    # Save adjacency for best
    if cmaes_result['best_params'] is not None:
        if A_ndp_best is not None:
            n_best = A_ndp_best.shape[0]
            df_adj = pd.DataFrame(
                A_ndp_best.astype(int), 
                index=neurons[:n_best], 
                columns=neurons[:n_best])
            gen_tag = f"gen{cmaes_result['best_generation']}"
            xlsx_path = output_dir / f"Adjacency_best_{gen_tag}.xlsx"
            df_adj.to_excel(xlsx_path, sheet_name="adjacency")

            edge_list = [(neurons[int(u)], neurons[int(v)]) for u, v in G_ndp_best.edges()]
            df_edges = pd.DataFrame(edge_list, columns=["pre", "post"])
            with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
                df_edges.to_excel(writer, sheet_name="edge_list", index=False)

            Results["best_adj"] = str(xlsx_path)


    output_path = output_dir / 'Results_NDP.yaml'

    with open(output_path, 'w') as f:
        yaml.dump(Results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    if cmaes_result['best_params'] is not None:
        np.save(output_dir / 'best_params.npy', cmaes_result['best_params'])

    return Results, output_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimize NDP parameters using CMA-ES")
    parser.add_argument("--config", type=str, default=str(CONFIG))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)

    log_path = output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            Results, output_path = main_run(config)
            print(f"\nResults saved to: {output_path}", flush=True)

            if wandb.run is not None:
                art = wandb.Artifact("ndp_outputs", type="results")
                art.add_file(str(output_path))  # Results_NDP.yaml

                output_dir = Path(output_path).parent
                best_params_path = output_dir / "best_params.npy"
                if best_params_path.exists():
                    art.add_file(str(best_params_path))

                wandb.log_artifact(art)

                # Store final scores in W&B summary
                wandb.summary["final/best_gen"] = Results["ndp_best"]["generation"]
                wandb.summary["final/f1"] = Results["ndp_best"]["train_metrics"]["f1"]
                wandb.summary["final/jaccard"] = Results["ndp_best"]["train_metrics"]["edge_jaccard"]
                wandb.summary["final/soft_edge_balanced_loss"] = Results["ndp_best"]["train_metrics"]["soft_edge_balanced_loss"]
                wandb.summary["final/auroc"] = Results["ndp_best"]["train_metrics"]["auroc"]

# Example usage:
# First of all: .venv\Scripts\activate

    # Then: python a00_main.py --config config.yaml
