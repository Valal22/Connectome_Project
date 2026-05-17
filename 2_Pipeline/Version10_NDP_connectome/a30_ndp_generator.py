from pathlib import Path
import yaml
import torch
import networkx as nx
import numpy as np
import torch
from numpy.random import default_rng
from scipy.stats import uniform
from scipy import sparse 
import pandas as pd
from scipy.special import expit



from pprint import pprint 



HERE = Path.cwd()  
DATASETS = HERE / "datasets"


# =====================================================================
# Initial graph generation
# =====================================================================
def generate_initial_graph(network_size, sparsity, undirected, seed=None):
    rng = default_rng(seed)
    
    # Keep generating until we get a connected graph
    max_attempts = 100
    for _ in range(max_attempts):
        # Generate random binary adjacency matrix
        rvs = uniform(loc=0, scale=1).rvs

        W = np.rint(sparse.random(network_size, network_size,
                                   density=sparsity, 
                                   data_rvs=rvs, 
                                   random_state=rng).toarray())
        
        # Check connectivity
        if undirected:
            G_check = nx.from_numpy_array(W, create_using=nx.Graph)
            if nx.is_connected(G_check):
                break
        else:
            G_check = nx.from_numpy_array(W, create_using=nx.DiGraph)
            if nx.is_weakly_connected(G_check):
                break
    
    if undirected:
        G = nx.from_numpy_array(W, create_using=nx.Graph)
    else:
        G = nx.from_numpy_array(W, create_using=nx.DiGraph)

    if G is None:
        raise RuntimeError("Failed to generate a connected initial graph in 100 attempts.")
    
    return G


# =====================================================================
# Embedding dimension calculator
# =====================================================================
def embeddings_dimensions(config):
    meta_mode = config["meta_embedding_mode"]
    mode = config["embedding_mode"]

    if mode == "fixed_only":
        fixed_D = int(config["fixed_embedding_size"])
        dynamic_D = 0
    elif mode == "dynamic_only":
        fixed_D = 0
        dynamic_D = int(config["dynamic_embedding_size"])
    else:
        raise ValueError(f"Unknown embedding_mode: {mode!r}")

    if fixed_D == 0 and dynamic_D == 0:
        raise ValueError(
            "At least one of fixed_embedding_size or dynamic_embedding_size must be > 0."
        )

    mode_D = fixed_D + dynamic_D

    if meta_mode == "class_only":
        if not config["class_emb"]:
            raise ValueError("meta_embedding_mode='class_only' requires config['class_emb']=True.")

        coords_D = 3 if config["coords_emb"] else 0
        transcr_D = int(config["genes"]) if config["transcr_emb"] else 0
        mode_D_fixed = coords_D + transcr_D

    elif meta_mode == "transc_only":
        if not config["transcr_emb"]:
            raise ValueError("meta_embedding_mode='transc_only' requires config['transcr_emb']=True.")

        coords_D = 3 if config["coords_emb"] else 0
        transcr_D = int(config["genes"])
        mode_D_fixed = coords_D + transcr_D

    elif meta_mode == "coords_only":
        if not config["coords_emb"]:
            raise ValueError("meta_embedding_mode='coords_only' requires config['coords_emb']=True.")

        coords_D = 3
        transcr_D = int(config["genes"]) if config["transcr_emb"] else 0
        mode_D_fixed = coords_D + transcr_D

    elif meta_mode == "hybrid":     # TO DO
        if not config["hybrid_emb"]:
            raise ValueError("meta_embedding_mode='hybrid' requires config['hybrid_emb']=True.")

        coords_D = 3 if config["coords_emb"] else 0
        transcr_D = int(config["genes"]) if config["transcr_emb"] else 0
        mode_D_fixed = coords_D + transcr_D

    else:
        raise ValueError(f"Unknown meta_embedding_mode: {meta_mode!r}")

    if config["use_fixed_feats"] != (mode_D_fixed > 0):
        raise ValueError(
            f"Inconsistent fixed-feature config: use_fixed_feats={config['use_fixed_feats']} "
            f"but mode_D_fixed={mode_D_fixed}."
        )

    return mode, fixed_D, dynamic_D, mode_D, mode_D_fixed
    







# =====================================================================
# MLP builder
# =====================================================================
def MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias):
    layers = []
    layers.append(torch.nn.Linear(input_dim, hidden_layers_dims[0], bias=bias))
    layers.append(activation)
    for i in range(1, len(hidden_layers_dims)):
        layers.append(torch.nn.Linear(hidden_layers_dims[i - 1], hidden_layers_dims[i], bias=bias))
        layers.append(activation)
    layers.append(torch.nn.Linear(hidden_layers_dims[-1], output_dim, bias=bias))
    if last_layer_activated:
        layers.append(activation)
    return torch.nn.Sequential(*layers).double()








# =====================================================================
# Message passing 
# =====================================================================
def propagate_features(growth_network_state, 
                       edge_network_state,
                       W, 
                       network_thinking_time, 
                       recurrent_activation_function, 
                       additive_update, 
                       feature_transformation_model=None,
                       config=None
                       ):

    mode = config["embedding_mode"]

    with torch.no_grad():
        growth_network_state = torch.tensor(growth_network_state, dtype=torch.float64)
        edge_network_state = torch.tensor(edge_network_state, dtype=torch.float64)
        W = torch.tensor(W, dtype=torch.float64)

        if config["GRAND_mp"]:
            if mode == "dynamic_only":
                # GRAND++ source term: save initial state
                h0 = edge_network_state.clone()

                # Source strength: evolved or from config
                # alpha=0 → pure diffusion (old behavior), alpha=1 → no diffusion
                alpha = float(config["mp_source_alpha"])

                if recurrent_activation_function == "tanh":
                    activation = torch.tanh
                else:
                    activation = None

                for step in range(network_thinking_time):

                    # Diffusion step
                    if additive_update:
                        h_diffused = edge_network_state + W.T @ edge_network_state
                    else:
                        h_diffused = W.T @ edge_network_state

                    # GRAND++ mixing: blend diffusion result with initial state
                    edge_network_state = (1.0 - alpha) * h_diffused + alpha * h0

                    if config["apply_activation_in_mp"] and activation is not None:
                        edge_network_state = activation(edge_network_state)

            elif mode == "fixed_only":
                # GRAND++ source term: save initial state
                h0 = growth_network_state.clone()

                # Source strength: evolved or from config
                # alpha=0 → pure diffusion (old behavior), alpha=1 → no diffusion
                alpha = float(config["mp_source_alpha"])

                if recurrent_activation_function == "tanh":
                    activation = torch.tanh
                else:
                    activation = None

                for step in range(network_thinking_time):

                    # Diffusion step
                    if additive_update:
                        h_diffused = growth_network_state + W.T @ growth_network_state
                    else:
                        h_diffused = W.T @ growth_network_state

                    # GRAND++ mixing: blend diffusion result with initial state 
                    growth_network_state = (1.0 - alpha) * h_diffused + alpha * h0

                    if config["apply_activation_in_mp"] and activation is not None:
                        growth_network_state = activation(growth_network_state)

        else:
            if recurrent_activation_function == "tanh":
                activation = torch.tanh
            else:
                activation = None

            for _ in range(network_thinking_time):
                if mode == "dynamic_only":
                    if additive_update:
                        edge_network_state = edge_network_state + W.T @ edge_network_state  


                    else:
                        edge_network_state = W.T @ edge_network_state 

                    if config["apply_activation_in_mp"] and activation is not None:
                        edge_network_state = activation(edge_network_state)

                elif mode == "fixed_only":
                    if additive_update:
                        growth_network_state = growth_network_state + W.T @ growth_network_state  


                    else:
                        growth_network_state = W.T @ growth_network_state

                    if config["apply_activation_in_mp"] and activation is not None:
                        growth_network_state = activation(growth_network_state)

        # Preserves directional diversity; controls magnitude (or not)
        if mode == "dynamic_only":
            # Worst (3)
            if config["L2_norm_mp"]:
                norms = torch.norm(edge_network_state, dim=1, keepdim=True)

                if torch.any(norms == 0):
                    idx = torch.nonzero(norms == 0, as_tuple=False)[0, 0].item()
                    raise ValueError(f"Zero L2 norm encountered in edge_network_state at row {idx}")

                edge_network_state = edge_network_state / norms

            # Worse (2)
            elif config["div_mp"]:
                norms = torch.norm(edge_network_state, dim=1, keepdim=True)

                if torch.any(norms == 0):
                    idx = torch.nonzero(norms == 0, as_tuple=False)[0, 0].item()
                    raise ValueError(f"Zero L2 norm encountered in edge_network_state at row {idx}")

                fixed_scale_divisor = float(config["fixed_scale_divisor"])

                edge_network_state = edge_network_state / fixed_scale_divisor

            # Best (1) 
            else:
                pass

        if mode == "fixed_only":
            # Worst (3)
            if config["L2_norm_mp"]:
                norms = torch.norm(growth_network_state, dim=1, keepdim=True)

                if torch.any(norms == 0):
                    idx = torch.nonzero(norms == 0, as_tuple=False)[0, 0].item()
                    raise ValueError(f"Zero L2 norm encountered in growth_network_state at row {idx}")

                growth_network_state = growth_network_state / norms

            # Worse (2)
            elif config["div_mp"]:
                norms = torch.norm(growth_network_state, dim=1, keepdim=True)

                if torch.any(norms == 0):
                    idx = torch.nonzero(norms == 0, as_tuple=False)[0, 0].item()
                    raise ValueError(f"Zero L2 norm encountered in growth_network_state at row {idx}")

                fixed_scale_divisor = float(config["fixed_scale_divisor"])

                growth_network_state = growth_network_state / fixed_scale_divisor

            # Best (1) 
            else:
                pass

    edge_network_state = edge_network_state.detach().numpy()
    growth_network_state = growth_network_state.detach().numpy()

    return edge_network_state, growth_network_state









# =====================================================================
# Prediction of new nodes
# =====================================================================
def predict_new_nodes(mlp_growth_model, growth_network_state, edge_network_state, config):
    new_nodes_predictions = []

    mode = config["embedding_mode"]

    if mode == "dynamic_only":
        with torch.no_grad():
            predictions_probabilities = mlp_growth_model(torch.tensor(edge_network_state, dtype=torch.float64)).detach().numpy()
            
            # # If debug:
            # print(f"[predict_new_nodes] raw output: {predictions_probabilities.ravel()}")

            new_nodes_predictions = (predictions_probabilities > 0).squeeze()
    

    elif mode == "fixed_only":
        with torch.no_grad():
            predictions_probabilities = mlp_growth_model(torch.tensor(growth_network_state, dtype=torch.float64)).detach().numpy()
            
            # # If debug:
            # print(f"[predict_new_nodes] raw output: {predictions_probabilities.ravel()}")

            new_nodes_predictions = (predictions_probabilities > 0).squeeze()


    return new_nodes_predictions



# =====================================================================
# Adding new nodes
# =====================================================================
def add_new_nodes(
    G,
    config,
    edge_network_state,
    growth_network_state,
    fixed_features, 
    new_nodes_predictions,
    rng=None,
    neurons_current=None,
    growth_cycle=None,
    n_emb=None,
    processed_fixed_feats_dict=None,
):


    current_graph_size = len(G)

    if n_emb is None:
        raise ValueError("n_emb must be provided to add_new_nodes")
    common_names = config["common_names"]

    if config["use_fixed_feats"] and processed_fixed_feats_dict is None:
        raise ValueError("processed_fixed_feats_dict must be provided when config['coords_fixed'] is True")
    
    if neurons_current is None:
        raise ValueError("neurons_current must be provided to add_new_nodes.")


    if config["default_growth"]:
        if current_graph_size >= len(common_names):
            born_nodes = []
            birth_events = []
            return G, growth_network_state, edge_network_state, fixed_features, neurons_current, born_nodes, birth_events

    else:
        target_network_size = int(config["target_network_size"])

        if current_graph_size > target_network_size:
            raise ValueError(
                f"Graph already has too many nodes: "
                f"{current_graph_size} > {target_network_size}"
            )

        if current_graph_size == target_network_size:
            born_nodes = []
            birth_events = []
            return G, growth_network_state, edge_network_state, fixed_features, neurons_current, born_nodes, birth_events
        
        remaining_nodes = target_network_size - current_graph_size

    # Get neighbors for each node BEFORE adding new nodes
    if len(G) == 1:
        neighbors = np.array([[0]])
    else: 
        neighbors = []
        for idx_node in range(len(G)):
            neighbors_idx = [n for n in nx.all_neighbors(G, idx_node)]
            neighbors_idx.append(idx_node)
            neighbors.append(np.unique(neighbors_idx))

    born_nodes = []
    birth_events = []

    if config["default_growth"]:
        new_nodes_predictions = new_nodes_predictions.reshape(-1)

        if not config["controlled_node_addition"]:
            growth_candidates = np.flatnonzero(new_nodes_predictions)

        # Add at most X new node(s) per growth cycle.
        else:
            growth_candidates = np.flatnonzero(new_nodes_predictions)
            max_nodes_per_cycle = int(config["max_new_nodes_per_cycle"])

            if growth_candidates.size > max_nodes_per_cycle:
                growth_candidates = growth_candidates[:max_nodes_per_cycle]

    else:
        if config["controlled_node_addition"]:
            max_nodes_per_cycle = int(config["max_new_nodes_per_cycle"])

            if max_nodes_per_cycle <= 0:
                raise ValueError(
                    f"max_new_nodes_per_cycle must be > 0, got {max_nodes_per_cycle}"
                )

            nodes_to_add_this_cycle = min(max_nodes_per_cycle, remaining_nodes)
        else:
            nodes_to_add_this_cycle = remaining_nodes

        if current_graph_size <= 0:
            raise ValueError(
                f"Cannot grow from an empty graph: current_graph_size={current_graph_size}"
            )

        growth_candidates = np.arange(nodes_to_add_this_cycle, dtype=int) % current_graph_size

    for idx_node in growth_candidates:
        if len(neighbors) != 0:
            # Add new node
            G.add_node(current_graph_size)

            born_nodes.append(current_graph_size)

            # Birth provenance (parent -> child) 
            parent_local_index = int(idx_node)
            child_local_index = int(current_graph_size)

            if "global_id" not in G.nodes[parent_local_index]:
                raise KeyError(f"Missing 'global_id' on parent node local_index={parent_local_index}")
            parent_global_id = int(G.nodes[parent_local_index]["global_id"])


            if 0 <= parent_global_id < len(common_names):
                parent_name = common_names[parent_global_id]
            else:
                parent_name = f"extra_{parent_global_id}"

            # Pick the next REAL neuron from the dataset pool
            available = config["_available_global_ids"]
            if len(available) > 0:
                gid = int(available.pop(0))
                child_name = common_names[gid]
                
            else:
                gid = int(current_graph_size)
                child_name = f"extra_{gid}"

            if child_name in config["n_emb"]:
                new_growth_row = config["n_emb"][child_name].reshape(1, -1)
                new_edge_row = new_growth_row.copy()
            else:
                new_growth_row = growth_network_state[parent_local_index].reshape(1, -1).copy()
                new_edge_row = edge_network_state[parent_local_index].reshape(1, -1).copy()

            growth_network_state = np.concatenate([growth_network_state, new_growth_row], axis=0)
            edge_network_state = np.concatenate([edge_network_state, new_edge_row], axis=0)

            if config["use_fixed_feats"]:
                if child_name in processed_fixed_feats_dict:
                    new_fixed_row = processed_fixed_feats_dict[child_name].reshape(1, -1)
                else:
                    new_fixed_row = fixed_features[parent_local_index].reshape(1, -1).copy()

                fixed_features = np.concatenate([fixed_features, new_fixed_row], axis=0)


            G.nodes[current_graph_size]["global_id"] = gid
            neurons_current.append(child_name)

            birth_events.append({
                "growth_cycle": int(growth_cycle),
                "parent_local_index": int(parent_local_index),
                "parent_global_id": int(parent_global_id),
                "parent_name": parent_name,
                "child_local_index": int(child_local_index),
                "child_global_id": int(gid),
                "child_name": child_name,
            })

            if rng is None:
                raise ValueError("rng must be provided to add_new_nodes.")

            current_graph_size += 1


    return (
        G,
        growth_network_state,
        edge_network_state,
        fixed_features,
        neurons_current,
        born_nodes,
        birth_events,
    )



# =====================================================================
# Edge wiring rule
# =====================================================================
def wiring_rule(
    G,
    edge_network_state,
    fixed_features, 
    mlp_edge_model,
    mlp_prune_model,
    rng,
    config,
    growth_cycle=None,    
    seed=None,         
):
    n_nodes = G.number_of_nodes()
 
    # IMPORTANT: rewire from scratch each cycle (otherwise edges accumulate and saturate).
    # MP has already happened upstream; this function only rebuilds the edge set.
    if config["pruning"]:
        if mlp_prune_model is None:
            raise ValueError("mlp_prune_model must be provided when config['pruning'] is True")

        existing_A = nx.to_numpy_array(
            G,
            nodelist=list(range(n_nodes)),
            dtype=bool,
        )
        np.fill_diagonal(existing_A, False)
    else:
        existing_A = np.zeros((n_nodes, n_nodes), dtype=bool)
        G.remove_edges_from(list(G.edges()))


    # Trivial-case shortcut: with 0 or 1 node there are no possible non-self edges.
    if n_nodes < 2:
        if bool(config["store_soft_P"]):
            G.graph["_soft_P"] = np.zeros((n_nodes, n_nodes), dtype=np.float64)
            G.graph["_final_logit"] = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float64)
        return G, edge_network_state

    # Per-node state fed to the edge MLP (concatenate fixed features if enabled).
    if config["use_fixed_feats"]:
        state = np.concatenate([edge_network_state, fixed_features], axis=1)
    else:
        state = edge_network_state
 
    D = state.shape[1]
 
    state_t = torch.from_numpy(np.ascontiguousarray(state, dtype=np.float64))
 
    expected_input_dim = 2 * D + (1 if config["pair_distance_feat"] else 0)

    if config["pair_distance_feat"]:
        chunk_size = int(config["edge_pair_source_chunk_size"])

        if chunk_size <= 0:
            raise ValueError(f"edge_pair_source_chunk_size must be > 0, got {chunk_size}")

        pair_dist_mean = float(config["_pair_dist_mean"])
        pair_dist_std = float(config["_pair_dist_std"])

        if pair_dist_std == 0.0:
            raise ValueError("config['_pair_dist_std'] is zero")

        coords_t = torch.from_numpy(
            np.ascontiguousarray(fixed_features[:, :3], dtype=np.float64)
        )


    final_logit_matrix = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float64)



    with torch.no_grad():
        for src_start in range(0, n_nodes, chunk_size):
            src_end = min(src_start + chunk_size, n_nodes)
            block_rows = src_end - src_start

            src_block = state_t[src_start:src_end]

            src = src_block.repeat_interleave(n_nodes, dim=0)
            dst = state_t.repeat(block_rows, 1)

            pair_parts = [src, dst]

            if config["pair_distance_feat"]:
                coords_src_block = coords_t[src_start:src_end]
                coords_src = coords_src_block.repeat_interleave(n_nodes, dim=0)
                coords_dst = coords_t.repeat(block_rows, 1)

                pair_dist = torch.linalg.vector_norm(
                    coords_src - coords_dst,
                    dim=1,
                    keepdim=True,
                )

                pair_dist = (pair_dist - pair_dist_mean) / pair_dist_std
                pair_parts.append(pair_dist)

            pair_input = torch.cat(pair_parts, dim=1)

            if pair_input.shape[1] != expected_input_dim:
                raise ValueError(
                    f"pair_input has dim {pair_input.shape[1]}, expected {expected_input_dim}"
                )

            logits_block_flat = mlp_edge_model(pair_input).squeeze(-1)
            logits_block = logits_block_flat.view(block_rows, n_nodes)

            final_logit_matrix[src_start:src_end, :] = (
                logits_block.numpy().astype(np.float64, copy=False)
            )

    np.fill_diagonal(final_logit_matrix, -np.inf)

    P_soft = expit(final_logit_matrix)
    np.fill_diagonal(P_soft, 0.0)
 

    if config["pruning"]:
        prune_logit_matrix = np.full((n_nodes, n_nodes), -np.inf, dtype=np.float64)

        with torch.no_grad():
            for src_start in range(0, n_nodes, chunk_size):
                src_end = min(src_start + chunk_size, n_nodes)
                block_rows = src_end - src_start

                src_block = state_t[src_start:src_end]
                src = src_block.repeat_interleave(n_nodes, dim=0)
                dst = state_t.repeat(block_rows, 1)

                pair_parts = [src, dst]

                if config["pair_distance_feat"]:
                    coords_src_block = coords_t[src_start:src_end]
                    coords_src = coords_src_block.repeat_interleave(n_nodes, dim=0)
                    coords_dst = coords_t.repeat(block_rows, 1)

                    pair_dist = torch.linalg.vector_norm(
                        coords_src - coords_dst,
                        dim=1,
                        keepdim=True,
                    )
                    pair_dist = (pair_dist - pair_dist_mean) / pair_dist_std
                    pair_parts.append(pair_dist)

                pair_input = torch.cat(pair_parts, dim=1)

                if pair_input.shape[1] != expected_input_dim:
                    raise ValueError(
                        f"prune pair_input has dim {pair_input.shape[1]}, expected {expected_input_dim}"
                    )

                prune_logits_block_flat = mlp_prune_model(pair_input).squeeze(-1)
                prune_logits_block = prune_logits_block_flat.view(block_rows, n_nodes)

                prune_logit_matrix[src_start:src_end, :] = (
                    prune_logits_block.numpy().astype(np.float64, copy=False)
                )

        np.fill_diagonal(prune_logit_matrix, -np.inf)
        prune_prob_matrix = expit(prune_logit_matrix)
        np.fill_diagonal(prune_prob_matrix, 0.0)

    if config["pruning"]:
        if config["prune_selection_mode"] == "threshold":
            prune_decision = prune_logit_matrix > float(config["prune_logit_threshold"])
        elif config["prune_selection_mode"] == "bernoulli":
            random_prune = rng.random((n_nodes, n_nodes))
            prune_decision = random_prune < prune_prob_matrix
        else:
            raise ValueError(
                f"Unsupported prune_selection_mode: {config['prune_selection_mode']}"
            )

        prune_mask = existing_A & prune_decision
        survive_mask = existing_A & ~prune_mask

        P_soft[survive_mask] = 1.0 - prune_prob_matrix[survive_mask]
        P_soft[prune_mask] = 0.0
        final_logit_matrix[prune_mask] = -np.inf
    else:
        prune_mask = np.zeros((n_nodes, n_nodes), dtype=bool)
        survive_mask = np.zeros((n_nodes, n_nodes), dtype=bool)


    # Store 
    if bool(config["store_soft_P"]):
        G.graph["_soft_P"] = P_soft
        G.graph["_final_logit"] = final_logit_matrix
 

    edge_logit_threshold = float(config["edge_logit_threshold"])
    use_logit_threshold_edges = bool(config["use_logit_threshold_edges"])

    candidate_mask = ~np.eye(n_nodes, dtype=bool)

    if config["pruning"]:
        candidate_mask &= ~survive_mask

        candidate_mask &= ~prune_mask

    if use_logit_threshold_edges:
        edge_mask = (final_logit_matrix > edge_logit_threshold) & candidate_mask
    else:
        random_edge = rng.random((n_nodes, n_nodes))
        edge_mask = (random_edge < P_soft) & candidate_mask

    if config["pruning"]:
        final_edge_mask = survive_mask | edge_mask
    else:
        final_edge_mask = edge_mask

    G.remove_edges_from(list(G.edges()))

    src_idx, dst_idx = np.where(final_edge_mask)
    if src_idx.size > 0:
        G.add_edges_from(zip(src_idx.tolist(), dst_idx.tolist()))

    return G, edge_network_state






# =====================================================================
# Main growth loop
# =====================================================================
def grow_network(evolved_params, config, seed=None, verbose=False):
    if seed is None:
        seed = config["seed"]

    np.random.seed(seed)
    torch.manual_seed(seed) # makes Torch ops reproducible
    rng = np.random.default_rng(seed) # makes NumPy draws reproducible

    # Create initial graph
    G = generate_initial_graph(
        network_size=int(config["initial_network_size"]),
        sparsity=float(config["initial_sparsity"]),
        undirected=bool(config["undirected"]),
        seed=seed,
    )

    common_names = config["common_names"]

    # Birth pool: dataset indices available beyond the evaluated subset 
    initN = config['initial_network_size']
    if initN < 0:
        raise ValueError(f"initial_network_size must be >= 0, got {initN}.")
    if initN > len(common_names):
        raise ValueError(
            f"initial_network_size ({initN}) exceeds number of available neurons (len(common_names)={len(common_names)})."
        )

    # Names for CURRENT graph nodes in local-index order.
    neurons_current = []
    for i in range(initN):
        neurons_current.append(common_names[i])
    
    # Birth pool should start after the neurons already present at init time
    config["_available_global_ids"] = list(range(initN, len(common_names)))

    # Assign global_id to existing nodes 0..(initN-1)
    for n in G.nodes():
        G.nodes[n]["global_id"] = n

    mode, fixed_D, dynamic_D, mode_D, mode_D_fixed = embeddings_dimensions(config)

    if "_static_node_features_prepared" not in config:
        raise KeyError("Missing config['_static_node_features_prepared']; call prepare_static_node_features_once(config) before grow_network.")

    if config["_static_node_features_prepared"] is not True:
        raise ValueError("config['_static_node_features_prepared'] must be True before grow_network.")

    n_emb = config["n_emb"]

    if config["use_fixed_feats"]:
        processed_fixed_feats_dict = config["processed_fixed_feats"]
    else:
        processed_fixed_feats_dict = None
    
    if config["default_growth"]:
        # Build MLPs
        mlp_growth_model = MLP(
            input_dim=mode_D,  
            output_dim=1,
            hidden_layers_dims=config['mlp_growth_hidden_layers_dims'],
            last_layer_activated=config['growth_model_last_layer_activated'],
            activation=torch.nn.Tanh(),
            bias=config['growth_model_bias']
        )
    else:
        pass

    # TO DO?    
    # if config["dynamic_only"]:
        # mlp_feature_transformation = None
        # if config['NN_transform_node_embedding_during_growth']:
        #     if dynamic_D == 0:
        #         raise ValueError(
        #             "NN_transform_node_embedding_during_growth=True requires dynamic_embedding_size > 0."
        #         )
            
        #     mlp_feature_transformation = MLP(
        #         # input_dim=config['node_embedding_size'],
        #         input_dim=mode_D,

        #         # output_dim=config['node_embedding_size'],
        #         output_dim=mode_D,

        #         hidden_layers_dims=config['mlp_embedding_transform_hidden_layers_dims'],
        #         last_layer_activated=config['transform_model_last_layer_activated'],
        #         activation=torch.nn.Tanh(),
        #         bias=config['transform_model_bias']
        #     )
        # TO DO

    edge_input_size = 2 * (mode_D + mode_D_fixed) + (1 if config["pair_distance_feat"] else 0)
    
    mlp_edge_model = MLP(
        input_dim=edge_input_size,
        output_dim=1,
        hidden_layers_dims=config['mlp_edge_hidden_layers_dims'],
        last_layer_activated=config['edge_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['edge_model_bias']
    )

    mlp_prune_model = None 
    if config["pruning"]:
        mlp_prune_model = MLP(
            input_dim=edge_input_size,
            output_dim=1,
            hidden_layers_dims=config["mlp_prune_hidden_layers_dims"],
            last_layer_activated=config["prune_model_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["prune_model_bias"],
        )

    initN = int(config['initial_network_size'])
    
    n1 = int(config['nb_params_coevolve_initial_embeddings']) if config["coevolve_initial_embeddings"] else 0
    
    n2 = n1 + config['nb_params_growth_model']

    n3 = n2 + config['nb_params_feature_transformation']  # end of transform MLP
    n4 = n3 + config['nb_params_edge_model']  # end of edge MLP

    if config["pruning"]: 
        n5 = n4 + config["nb_params_prune_model"]

        torch.nn.utils.vector_to_parameters(
            torch.tensor(evolved_params[n4:n5], dtype=torch.float64, requires_grad=False),
            mlp_prune_model.parameters(),
        )

    growth_network_state = np.zeros((initN, mode_D), dtype=np.float64)
    edge_network_state = np.zeros((initN, mode_D), dtype=np.float64)
    fixed_features = np.zeros((initN, mode_D_fixed), dtype=np.float64)


    if config["initial_embeddings_from_class"]:
        if initN > len(common_names):
            raise ValueError(f"initN={initN} exceeds len(common_names)={len(common_names)}")

        for local_i in range(initN):
            gid = local_i
            name = common_names[gid]

            if name not in n_emb:
                raise KeyError(f"Neuron '{name}' missing from full_neuron_class_names.")

            growth_network_state[local_i] = n_emb[name].reshape(-1)
            edge_network_state[local_i] = n_emb[name].reshape(-1)

            if config["use_fixed_feats"]:
                fixed_features[local_i] = processed_fixed_feats_dict[name].reshape(-1)
    
    # TO DO
    # elif  coordinates
    # elif transcr 

    # Default: 
    # else:
        # Look at the original files

    if config["default_growth"]:
        torch.nn.utils.vector_to_parameters(
            torch.tensor(evolved_params[n1:n2], dtype=torch.float64, requires_grad=False),
            mlp_growth_model.parameters())
    else:
        pass

    # TO DO?
    # if config['NN_transform_node_embedding_during_growth']:
    #     torch.nn.utils.vector_to_parameters(
    #         torch.tensor(evolved_params[n2:n3], dtype=torch.float64, requires_grad=False),
    #         mlp_feature_transformation.parameters())

    torch.nn.utils.vector_to_parameters(
        torch.tensor(evolved_params[n3:n4], dtype=torch.float64, requires_grad=False),
        mlp_edge_model.parameters())

    for growth_cycle in range(config['number_of_growth_cycles']):
        if config["default_thinking_time"]:
            try:
                if config['undirected']:
                    diameter = nx.diameter(G)
                    # Debug
                    # print(f"[cycle {growth_cycle}] diameter (try) = {diameter}")
                    # Debug
                else:
                    diameter = nx.diameter(G.to_undirected())
            except (nx.NetworkXError, ValueError):
                diameter = max(1, int(np.sqrt(len(G))))
                # Debug
                # print(f"[cycle {growth_cycle}] diameter (except) = {diameter}")
                # Debug
            
            network_thinking_time = diameter + config['network_thinking_time_extra_growth']

        else:   
            # Another way for thinking time
            if "network_thinking_time" not in config:
                raise KeyError("Missing required config key: 'network_thinking_time' (int >= 1)")

            network_thinking_time = int(config["network_thinking_time"])
            if network_thinking_time < 0:
                raise ValueError(f"config['network_thinking_time'] must be >= 0, got {network_thinking_time}")
            
            network_thinking_time += int(config["network_thinking_time_extra_growth"])

        if verbose:
            print("\n" + "=" * 80)
            print(f"GROWTH CYCLE {growth_cycle}")
            print("=" * 80)

            print(f"nodes present: {G.number_of_nodes()}")
            print(f"edges present before MP: {G.number_of_edges()}")

            node_debug = []
            growth_state_by_name = {}
            edge_state_by_name = {}
            
            for local_idx in G.nodes():
                if "global_id" not in G.nodes[local_idx]:
                    raise KeyError(f"Missing 'global_id' on node local_idx={local_idx}")

                global_id = int(G.nodes[local_idx]["global_id"])

                if 0 <= global_id < len(common_names):
                    neuron_name = common_names[global_id]
                else:
                    neuron_name = f"extra_{global_id}"

                node_debug.append(
                    {
                        "local_idx": int(local_idx),
                        "global_id": int(global_id),
                        "name": neuron_name,
                    }
                )

                growth_state_by_name[neuron_name] = growth_network_state[local_idx].tolist()
                edge_state_by_name[neuron_name] = edge_network_state[local_idx].tolist()


            print(f"neuron(s) present: {node_debug}")

            print("growth_network_state BEFORE mp BY NAME:")
            print(growth_state_by_name)

            print("edge_network_state BEFORE mp BY NAME:")
            print(edge_state_by_name)



        # Propagate features
        edge_network_state, growth_network_state = propagate_features(
            growth_network_state=growth_network_state,
            edge_network_state=edge_network_state,            
            W=nx.to_numpy_array(G),       
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config['recurrent_activation_function'],
            additive_update=config['additive_update'],
            config=config
        )

        # # If debug:
        # # Propagate features
        # network_state, edge_network_state, growth_network_state  = propagate_features(
        #     growth_network_state=growth_network_state,
        #     edge_network_state=edge_network_state,
        #     W=nx.to_numpy_array(G),       
        #     network_thinking_time=network_thinking_time,
        #     recurrent_activation_function=config['recurrent_activation_function'],
        #     additive_update=config['additive_update'],
        #     # feature_transformation_model=mlp_feature_transformation,
        #     # debug_trace=debug_trace,
        #     #debug_node_idx=debug_node_idx,
        #     config=config
        # )
        # # end debug

        if verbose:
            growth_state_by_name = {}
            edge_state_by_name = {}

            for local_idx in G.nodes():
                if "global_id" not in G.nodes[local_idx]:
                    raise KeyError(f"Missing 'global_id' on node local_idx={local_idx}")

                global_id = int(G.nodes[local_idx]["global_id"])

                if 0 <= global_id < len(common_names):
                    neuron_name = common_names[global_id]
                else:
                    neuron_name = f"extra_{global_id}"

                growth_state_by_name[neuron_name] = growth_network_state[local_idx].tolist()
                edge_state_by_name[neuron_name] = edge_network_state[local_idx].tolist()

            print("growth_network_state AFTER mp BY NAME:")
            print(growth_state_by_name)

            print("edge_network_state AFTER mp BY NAME:")
            print(edge_state_by_name)

            # # If debug:
            # print(f"growth_network_state AFTER MP:\n{growth_network_state}")
            # print(f"edge_network_state AFTER MP:\n{edge_network_state}")
            # # end debug

        if config["default_growth"]:
            new_nodes_predictions = predict_new_nodes(mlp_growth_model, growth_network_state, edge_network_state, config)

        else:        
            new_nodes_predictions = np.ones(growth_network_state.shape[0], dtype=bool)

        predictions_array = np.asarray(new_nodes_predictions, dtype=bool).reshape(-1)


        # # If debug:
        # if verbose:
        #     prediction_debug = []

        #     for local_idx, prediction in enumerate(predictions_array):
        #         if local_idx not in G.nodes:
        #             raise KeyError(f"Prediction exists for missing local_idx={local_idx}")

        #         if "global_id" not in G.nodes[local_idx]:
        #             raise KeyError(f"Missing 'global_id' on node local_idx={local_idx}")

        #         global_id = int(G.nodes[local_idx]["global_id"])

        #         if 0 <= global_id < len(common_names):
        #             neuron_name = common_names[global_id]
        #         else:
        #             neuron_name = f"extra_{global_id}"

        #         prediction_debug.append(
        #             {
        #                 "local_idx": int(local_idx),
        #                 "global_id": int(global_id),
        #                 "name": neuron_name,
        #                 "spawn": bool(prediction),
        #             }
        #         )

        #     print(f"Predictions T/F from neurons: {prediction_debug}")
        #     print(f"Total predicted newborn: {int(predictions_array.sum())}")
        # # end debug 





        # Add new nodes
        G, growth_network_state, edge_network_state, fixed_features, neurons_current, born_nodes, birth_events = add_new_nodes(
            G=G,
            config=config,
            growth_network_state=growth_network_state,
            edge_network_state=edge_network_state,
            fixed_features=fixed_features,
            new_nodes_predictions=new_nodes_predictions,
            rng=rng,
            neurons_current=neurons_current,
            growth_cycle=growth_cycle,
            n_emb=n_emb,
            processed_fixed_feats_dict=processed_fixed_feats_dict if config["use_fixed_feats"] else None,
        )


        if verbose:
            newborn_debug = []

            for event in birth_events:
                child_local_index = int(event["child_local_index"])
                child_name = event["child_name"]

                newborn_debug.append(
                    {
                        "growth_cycle": int(event["growth_cycle"]),
                        "parent_name": event["parent_name"],
                        "child_local_index": child_local_index,
                        "child_global_id": int(event["child_global_id"]),
                        "child_name": child_name,
                        "growth_embedding": growth_network_state[child_local_index].tolist(),
                        "edge_embedding": edge_network_state[child_local_index].tolist(),
                    }
                )

            print("NEWBORN NODES WITH EMBEDDINGS:")
            print(newborn_debug)

            fixed_features_by_name = {}

            for local_idx in G.nodes():
                if "global_id" not in G.nodes[local_idx]:
                    raise KeyError(f"Missing 'global_id' on node local_idx={local_idx}")

                global_id = int(G.nodes[local_idx]["global_id"])

                if 0 <= global_id < len(common_names):
                    neuron_name = common_names[global_id]
                else:
                    neuron_name = f"extra_{global_id}"

                fixed_features_by_name[neuron_name] = fixed_features[local_idx].tolist()

            print("fixed_features AFTER add_new_nodes BY NAME:")
            print(fixed_features_by_name)


        # # If debug:
        # if verbose:
        #     print(f"born_nodes local idx: {born_nodes}")
        #     print(f"birth_events: {birth_events}")
        #     print(f"nodes after add_new_nodes: {G.number_of_nodes()}")
        #     print(f"edges before wiring rule: {G.number_of_edges()}")
        #     print(f"edges before wiring rule list: {list(G.edges())}")
        # # end debug




        # Wiring rule
        edges_before_wiring = set(G.edges())
        
        G, edge_network_state = wiring_rule(
            G=G,
            edge_network_state=edge_network_state,
            fixed_features=fixed_features,
            mlp_edge_model=mlp_edge_model,
            mlp_prune_model=mlp_prune_model,
            rng=rng,
            config=config,
            growth_cycle=growth_cycle,                  
            seed=seed,  
        )

        
        if verbose:
            fixed_features_by_name = {}

            for local_idx in G.nodes():
                if "global_id" not in G.nodes[local_idx]:
                    raise KeyError(f"Missing 'global_id' on node local_idx={local_idx}")

                global_id = int(G.nodes[local_idx]["global_id"])

                if 0 <= global_id < len(common_names):
                    neuron_name = common_names[global_id]
                else:
                    neuron_name = f"extra_{global_id}"

                fixed_features_by_name[neuron_name] = fixed_features[local_idx].tolist()

            print("fixed_features AFTER wiring_rule BY NAME:")
            print(fixed_features_by_name)

    return G, edge_network_state, growth_network_state

            


# =====================================================================
# NDP parameter counting
# =====================================================================
def count_ndp_parameters(config):
    mode, fixed_D, dynamic_D, mode_D, mode_D_fixed = embeddings_dimensions(config)

    if config["coevolve_initial_embeddings"]:
        raise ValueError(
            "This split-embedding implementation requires coevolve_initial_embeddings=False."
        )

    nb_params_init_emb = 0

    if config["default_growth"]:
        mlp_growth = MLP(
            input_dim=mode_D,
            output_dim=1,
            hidden_layers_dims=config['mlp_growth_hidden_layers_dims'],
            last_layer_activated=config['growth_model_last_layer_activated'],
            activation=torch.nn.Tanh(),
            bias=config['growth_model_bias']
        )
        nb_params_growth = int(torch.nn.utils.parameters_to_vector(mlp_growth.parameters()).shape[0])

    else:
        nb_params_growth = 0


    nb_params_transform = 0
    # TO DO?
    # if config['NN_transform_node_embedding_during_growth']:
    #     if dynamic_D == 0:
    #         raise ValueError(
    #             "NN_transform_node_embedding_during_growth=True requires dynamic_embedding_size > 0."
    #         )

        # mlp_transform = MLP(
        #     input_dim=dynamic_D,
        #     output_dim=dynamic_D,
        #     hidden_layers_dims=config['mlp_embedding_transform_hidden_layers_dims'],
        #     last_layer_activated=config['transform_model_last_layer_activated'],
        #     activation=torch.nn.Tanh(),
        #     bias=config['transform_model_bias']
        # )
        # nb_params_transform = int(torch.nn.utils.parameters_to_vector(mlp_transform.parameters()).shape[0])

    mlp_edge = MLP(
        input_dim=2 * (mode_D + mode_D_fixed) + (1 if config["pair_distance_feat"] else 0),
        output_dim=1,
        hidden_layers_dims=config['mlp_edge_hidden_layers_dims'],
        last_layer_activated=config['edge_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['edge_model_bias']
    )
    nb_params_edge = int(torch.nn.utils.parameters_to_vector(mlp_edge.parameters()).shape[0])

    mlp_prune = None
    if config["pruning"]:
        mlp_prune = MLP(
            input_dim=2 * (mode_D + mode_D_fixed) + (1 if config["pair_distance_feat"] else 0),
            output_dim=1,
            hidden_layers_dims=config["mlp_prune_hidden_layers_dims"],
            last_layer_activated=config["prune_model_last_layer_activated"],
            activation=torch.nn.Tanh(),
            bias=config["prune_model_bias"],
        )
    
        nb_params_prune = int(torch.nn.utils.parameters_to_vector(mlp_prune.parameters()).shape[0])
    else:
        nb_params_prune = int(0) 

    config['nb_params_coevolve_initial_embeddings'] = int(nb_params_init_emb)
    config['nb_params_growth_model'] = int(nb_params_growth)
    config['nb_params_feature_transformation'] = int(nb_params_transform)
    config['nb_params_edge_model'] = int(nb_params_edge)
    config["nb_params_prune_model"] = int(nb_params_prune)
    
    total_params = (
        nb_params_init_emb
        + nb_params_growth
        + nb_params_edge
        + nb_params_prune
    )

    return int(total_params)
