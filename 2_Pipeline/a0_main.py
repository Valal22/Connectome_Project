import argparse
from pathlib import Path

from a1_utils_data_loading import load_hermaphrodite_chemical_connectome
from a2_utils_eval_metrics import adjacency_to_digraph, compute_eval_metrics
from a3_models_rand_generator import generate_er_random_graph
from a4_models_train_metr import train_metrics, print_subgraph

# Path to the file shipped inside this repo
HERE = Path(__file__).resolve().parent
DEFAULT_FILE = HERE / "SI_5_Connectome_adjacency_matrices_corrected_July_2020.xlsx"



###############
# Pipeline
###############

def run_rand_pipeline(file_path, num_epochs, p, seed):
    # 1. Load dataset
    A_target, neurons = load_hermaphrodite_chemical_connectome(file_path)
    N = A_target.shape[0]   # number of neurons
    print("#######")
    print("Results:")
    print(f"Loaded connectome with {N} neurons and {A_target.sum()} directed edges (binarized).")


    # 2. Target graph metrics
    G_target = adjacency_to_digraph(A_target, neurons)
    target_metrics = compute_eval_metrics(G_target)

    print("\nTarget graph metrics:")
    #print("avg_clustering           = %.4f" % target_metrics.avg_clustering)
    print("avg_shortest_path_length = %.4f. It seems to match Varhsney (TO CHECK)" % target_metrics.avg_shortest_path_length)
    #print("small_world        = %.4f" % target_metrics.small_world)
    print("num_hubs (top 10%%)       = %d. It is NOT matching Towlson (TO CHECK)" % len(target_metrics.hub_neurons))



    # 3. Training loop using simple Erdos–Renyi generator
    for epoch in range(1, num_epochs + 1):
        A_model, G_model = generate_er_random_graph(N, p=p, seed=12 + epoch)

        num_nodes = G_model.number_of_nodes()
        num_edges = G_model.number_of_edges()

        # Training metrics
        metrics = train_metrics(A_target, A_model)

        print(f"\nEpoch {epoch}")
        print("  [full graph] metrics (over flattened adjacency):")
        print(f"    n_entries            = {metrics['n_entries']}")
        print(f"    hamming_raw          = {metrics['hamming_raw']} / {metrics['n_entries']}")
        print(f"    hamming_norm         = {metrics['hamming_norm']:.4f}")
        print(f"    adjacency_similarity = {metrics['adjacency_similarity']:.4f}")
        print(f"    edge_jaccard         = {metrics['edge_jaccard']:.4f}")
        print(f"    precision            = {metrics['precision']:.4f}")
        print(f"    recall               = {metrics['recall']:.4f}")
        print(f"    f1                   = {metrics['f1']:.4f}")

        # Small subgraph inspection
        print_subgraph(A_target, A_model, neurons)

        # Evaluation metrics on the generated graph
        model_metrics = compute_eval_metrics(G_model)

        print("\nEpoch %d (model graph metrics)" % epoch)
        #print("avg_clustering(model)      = %.4f" % model_metrics.avg_clustering)
        print("avg_path_len(model)        = %.4f" % model_metrics.avg_shortest_path_length)
        #print("small_world_sigma(model)   = %.4f" % model_metrics.small_world_sigma)
        print("num_hubs (top 10%%)        = %d" % len(model_metrics.hub_neurons))
        print("num_nodes(model)           = %d" % num_nodes)
        print("num_edges(model)           = %d" % num_edges)



##################
# CLI entrypoint
##################

def parse_args():
    parser = argparse.ArgumentParser(description="rand connectome training pipeline")
    parser.add_argument(
        "--file-path",
        type=str,
        required=False,
        default=str(DEFAULT_FILE),
        help="Path to the connectome dataset file (default is the Cook herm chemical dataset).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of epochs for the rand ER training loop.",
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.0413,
        help="Edge probability for the Erdos-Renyi random graph.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12,
        help="Base random seed for the Erdos-Renyi random graph generator.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_rand_pipeline(
        file_path=args.file_path,
        num_epochs=args.num_epochs,
        p=args.p,
        seed=args.seed
    )
