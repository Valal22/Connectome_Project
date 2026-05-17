from pathlib import Path
import yaml
import pandas as pd
import networkx as nx
import numpy as np
import pickle 

from a10_data_load import load_witv, adj_in_birth_t, from_csv

HERE = Path.cwd()
DATASETS = HERE / "datasets"
CONFIG = HERE / "config.yaml"
ADJ_MATRIX = DATASETS / "synapse_count_matrices.xlsx"
GRAPH = HERE / "best_graph.pkl"

def load_pickle_graph(graph_path):
    with open(graph_path, "rb") as f:
        return pickle.load(f)

def adjacency_to_digraph(A_target, common_names):
    G = nx.DiGraph()
    G.add_nodes_from(common_names)
    N = A_target.shape[0]
    for i in range(N):
        for j in range(N):
            if A_target[i, j]:
                G.add_edge(common_names[i], common_names[j])
    return G


def compute_graph_statistics(G):
    is_dir = G.is_directed()
    N = G.number_of_nodes()
    M = G.number_of_edges()
    density = nx.density(G)

    if N == 0:
        return {
            "num_nodes": 0, "num_edges": 0, "density": 0,
            "degree_mean": 0, "degree_std": 0, "degree_min": 0, "degree_max": 0,
            "clustering_coefficient": 0, "num_connected_components": 0,
            "largest_cc_size": 0, "largest_cc_fraction": 0,
        }

    degrees = [d for _, d in G.degree()]
    cc = nx.average_clustering(G.to_undirected()) if is_dir else nx.average_clustering(G)

    if is_dir:
        n_cc = nx.number_weakly_connected_components(G)
        largest = max(nx.weakly_connected_components(G), key=len)
    else:
        n_cc = nx.number_connected_components(G)
        largest = max(nx.connected_components(G), key=len)

    return {
        "num_nodes": N, "num_edges": M, "density": density,
        "degree_mean": float(np.mean(degrees)),
        "degree_std": float(np.std(degrees)),
        "degree_min": int(np.min(degrees)),
        "degree_max": int(np.max(degrees)),
        "clustering_coefficient": cc,
        "num_connected_components": n_cc,
        "largest_cc_size": len(largest),
        "largest_cc_fraction": len(largest) / N,
    }



def compute_reciprocity(G):
    if not G.is_directed():
        return float("nan")

    num_edges = G.number_of_edges()
    if num_edges == 0:
        return 0.0

    reciprocal = sum(1 for u, v in G.edges() if G.has_edge(v, u))
    return reciprocal / num_edges

def clustering_stats(G):
    G_undir = G.to_undirected() if G.is_directed() else G
    cc = nx.clustering(G_undir)
    vals = list(cc.values())

    return {
        "mean": np.mean(vals),
        "std": np.std(vals),
        "median": np.median(vals),
        "values": vals,
    }


def small_worldness(G, n_random=10, seed=42):
    G_undir = G.to_undirected() if G.is_directed() else G
    if not nx.is_connected(G_undir):
        G_undir = G_undir.subgraph(max(nx.connected_components(G_undir), key=len)).copy()

    C = nx.transitivity(G_undir)
    L = nx.average_shortest_path_length(G_undir)

    rng = np.random.RandomState(seed)
    C_rands, L_rands = [], []
    n, m = G_undir.number_of_nodes(), G_undir.number_of_edges()
    for _ in range(n_random):
        R = nx.gnm_random_graph(n, m, seed=int(rng.randint(1e9)))
        if not nx.is_connected(R):
            R = R.subgraph(max(nx.connected_components(R), key=len)).copy()
        C_rands.append(nx.transitivity(R))
        L_rands.append(nx.average_shortest_path_length(R))

    C_rand = np.mean(C_rands)
    L_rand = np.mean(L_rands)

    sigma = (C / C_rand) / (L / L_rand) if C_rand > 0 and L_rand > 0 else float("nan")
    return {
        "sigma": sigma,
        "C": C,
        "L": L,
        "C_rand": C_rand,
        "L_rand": L_rand,
    }

def count_triadic_motifs(G):
    if not G.is_directed():
        return {}
    census = nx.triadic_census(G)
    return census



def compute_eval_metrics(G, verbose=False):
    stats = compute_graph_statistics(G)

    G_undir = G.to_undirected() if G.is_directed() else G

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    if verbose:
        print(f"Graph: {nodes} nodes, {edges} edges")

    degrees = np.array([d for _, d in G.degree()], dtype=float)

    if G.is_directed():
        in_degrees = np.array([d for _, d in G.in_degree()], dtype=float)
        out_degrees = np.array([d for _, d in G.out_degree()], dtype=float)
    else:
        in_degrees = np.array([], dtype=float)
        out_degrees = np.array([], dtype=float)

    cl = clustering_stats(G)
    sw = small_worldness(G, n_random=10, seed=42)

    trans = nx.transitivity(G_undir)
    recipr = compute_reciprocity(G)

    if nodes > 0:
        degree_threshold = np.percentile(degrees, 95)
        hub_neurons = [n for n, d in G.degree() if d >= degree_threshold]
        n_hubs = len(hub_neurons)
    else:
        degree_threshold = float("nan")
        hub_neurons = []
        n_hubs = 0

    if G.is_directed():
        wcc = list(nx.weakly_connected_components(G))
        scc = list(nx.strongly_connected_components(G))

        num_weakly_connected = len(wcc)
        largest_wcc = max(len(c) for c in wcc) if wcc else 0

        num_strongly_connected = len(scc)
        largest_scc = max(len(c) for c in scc) if scc else 0
    else:
        cc = list(nx.connected_components(G_undir))

        num_weakly_connected = float("nan")
        largest_wcc = float("nan")

        num_strongly_connected = len(cc)
        largest_scc = max(len(c) for c in cc) if cc else 0

    if G.is_directed():
        triadic_census = count_triadic_motifs(G)
    else:
        triadic_census = {}

    return {
        "nodes": int(nodes),
        "edges": int(edges),
        "is_directed": G.is_directed(),
        "density": float(nx.density(G)),

        "degree_mean": stats["degree_mean"],
        "degree_std": stats["degree_std"],
        "degree_min": stats["degree_min"],
        "degree_max": stats["degree_max"],

        "in_degree_mean": float(np.mean(in_degrees)) if in_degrees.size > 0 else float("nan"),
        "in_degree_std": float(np.std(in_degrees)) if in_degrees.size > 0 else float("nan"),
        "out_degree_mean": float(np.mean(out_degrees)) if out_degrees.size > 0 else float("nan"),
        "out_degree_std": float(np.std(out_degrees)) if out_degrees.size > 0 else float("nan"),

        "num_connected_components": stats["num_connected_components"],
        "largest_cc_size": stats["largest_cc_size"],
        "largest_cc_fraction": stats["largest_cc_fraction"],
        "num_weakly_connected": num_weakly_connected,
        "largest_wcc": largest_wcc,
        "num_strongly_connected": num_strongly_connected,
        "largest_scc": largest_scc,

        "clustering_coefficient": float(stats["clustering_coefficient"]),
        "avg_clustering": float(cl["mean"]),
        "clustering_std": float(cl["std"]),
        "clustering_median": float(cl["median"]),

        "transitivity": float(trans), # (global clustering)

        "small_world_coeff": float(sw["sigma"]),
        "avg_path_length": float(sw["L"]),
        "C": float(sw["C"]),
        "L": float(sw["L"]),
        "C_rand": float(sw["C_rand"]),
        "L_rand": float(sw["L_rand"]),

        "reciprocity": float(recipr),

        # TO FIX
        "n_hubs": int(n_hubs),
        "hub_neurons": hub_neurons,
        "degree_threshold_95": float(degree_threshold),
        # TO FIX
        
        "triadic_census": triadic_census,
    }






def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)



def print_graph_diagnostics(title, G):
    self_loops = list(nx.selfloop_edges(G))

    print(f"\n{title}")
    print(f"nodes: {G.number_of_nodes()}")
    print(f"edges including self-loops: {G.number_of_edges()}")
    print(f"self-loops: {len(self_loops)}")

    if len(self_loops) > 0:
        print(f"self-loop nodes: {[u for u, _ in self_loops]}")


def print_matrix_diagnostics(title, A_target, common_names):
    diag = np.diag(A_target)
    diag_idx = np.where(diag != 0)[0].tolist()
    diag_names = [common_names[i] for i in diag_idx]

    print(f"\n{title}")
    print(f"shape: {A_target.shape}")
    print(f"edges including diagonal: {int(A_target.sum())}")
    print(f"diagonal sum: {int(diag.sum())}")
    print(f"diagonal nonzero indices: {diag_idx}")
    print(f"diagonal nonzero names: {diag_names}")

    A_no_self = A_target.copy()
    np.fill_diagonal(A_no_self, 0)

    print(f"edges excluding diagonal: {int(A_no_self.sum())}")



def load_target_graphs(config):
    common_names, A_target = load_witv(
        file_path=ADJ_MATRIX,
        sheet_name=config["sheet_name"],
    )

    neurons_list, _, _ = from_csv(DATASETS / config["file_path"])

    common_names, A_target = adj_in_birth_t(
        common_names=common_names,
        A_target=A_target,
        neurons=neurons_list,
    )

    A_target_no_self = A_target.copy()
    np.fill_diagonal(A_target_no_self, 0)

    G_target_full = adjacency_to_digraph(A_target_no_self, common_names)

    if config["eval_first_n"] is not None:
        k = int(config["eval_first_n"])
        common_names_eval = common_names[:k]
        A_target_eval = A_target_no_self[:k, :k]
    else:
        common_names_eval = common_names
        A_target_eval = A_target_no_self

    G_target_sliced = adjacency_to_digraph(A_target_eval, common_names_eval)

    return {
        "common_names": common_names,
        "A_target_no_self": A_target_no_self,
        "common_names_eval": common_names_eval,
        "A_target_eval": A_target_eval,
        "G_target_full": G_target_full,
        "G_target_sliced": G_target_sliced,
    }



def print_metrics(title, metrics):
    print(f"\n{title}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


def main():
    config = load_config(CONFIG)

    G_model = load_pickle_graph(GRAPH)
    target_data = load_target_graphs(config)

    print_graph_diagnostics("Target full connectome before SCC:", target_data["G_target_full"])

    metrics_target_full = compute_eval_metrics(
        target_data["G_target_full"],
        verbose=False,
    )
    metrics_target_sliced = compute_eval_metrics(
        target_data["G_target_sliced"],
        verbose=False,
    )
    metrics_model = compute_eval_metrics(
        G_model,
        verbose=False,
    )

    print_metrics("Target full connectome metrics:", metrics_target_full)
    print_metrics("Target sliced connectome metrics:", metrics_target_sliced)
    print_metrics("Model graph metrics:", metrics_model)

    return metrics_target_full, metrics_target_sliced, metrics_model


if __name__ == "__main__":
    main()

# Example usage:
# First of all: .venv\Scripts\activate
    # Then: python eval.py