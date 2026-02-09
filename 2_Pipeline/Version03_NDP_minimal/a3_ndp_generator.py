
import time
import numpy as np
import torch
import networkx as nx
from scipy import sparse
from scipy.stats import uniform
from numpy.random import default_rng


import os
import matplotlib.pyplot as plt


# Disable gradients globally for inference
torch.autograd.set_grad_enabled(False)


def _render_frame(G, config, out_png, title="", highlight_node=None, highlight_edge=None, pos_cache=None):
    if pos_cache is None:
        pos_cache = nx.spring_layout(G, seed=0)
    else:
        # only assign positions to NEW nodes
        new_nodes = [n for n in G.nodes() if n not in pos_cache]
        if new_nodes:
            new_pos = nx.spring_layout(G, seed=0)
            for n in new_nodes:
                pos_cache[n] = new_pos[n]

    plt.figure(figsize=(6, 6), dpi=150)
    plt.axis("off")
    plt.title(title)

    # draw nodes
    node_colors = []
    for n in G.nodes():
        node_colors.append("tab:red" if (highlight_node is not None and n == highlight_node) else "tab:blue")

    nx.draw_networkx_nodes(G, pos_cache, node_size=140, node_color=node_colors)
    nx.draw_networkx_edges(G, pos_cache, alpha=0.35, arrows=nx.is_directed(G))


    neurons = config["neurons"]
    if neurons is not None and len(neurons) > 0:
        labels = {}
        for n in G.nodes():
            if n < len(neurons):
                labels[n] = f"{neurons[n]} ({n})"  
            else:
                labels[n] = str(n)
    else:
        labels = {n: str(n) for n in G.nodes()}

    nx.draw_networkx_labels(G, pos_cache, labels=labels, font_size=6)


    # highlight edge if requested
    if highlight_edge is not None:
        u, v = highlight_edge
        if G.has_edge(u, v):
            nx.draw_networkx_edges(G, pos_cache, edgelist=[(u, v)], width=2.5, alpha=0.9, arrows=nx.is_directed(G))

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png)
    plt.close()
    return pos_cache


def MLP(input_dim, output_dim, hidden_layers_dims, activation, last_layer_activated, bias):
    """
    Creates a multi-layer perceptron.

    Args:
        input_dim: Input dimension
        output_dim: Output dimension
        hidden_layers_dims: List of hidden layer sizes
        activation: Activation function (e.g., torch.nn.Tanh())
        last_layer_activated: Whether to apply activation to output
        bias: Whether to use bias terms

    Returns:
        torch.nn.Sequential: The MLP model
    """
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


def generate_initial_graph(network_size, sparsity, undirected, seed=None):
    """
    Generates a random initial graph.

    Args:
        network_size: Number of nodes
        sparsity: Edge density (0-1)
        undirected: If True, creates undirected graph
        seed: Random seed

    Returns:
        G: networkx Graph or DiGraph
    """
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

    return G


def propagate_features(network_state, W, network_thinking_time, recurrent_activation_function, 
                       additive_update, feature_transformation_model=None):
    """
    Propagates node features through the network.

    Args:
        network_state: (N, embedding_dim) array of node embeddings
        W: (N, N) adjacency matrix
        network_thinking_time: Number of propagation steps
        recurrent_activation_function: 'tanh' or None
        additive_update: If True, add propagated values to current state
        feature_transformation_model: Optional MLP to transform embeddings

    Returns:
        Updated network_state array
    """
    with torch.no_grad():
        network_state = torch.tensor(network_state, dtype=torch.float64)
                
        W = torch.tensor(W, dtype=torch.float64)

        if recurrent_activation_function == "tanh":
            activation = torch.tanh
        else:
            activation = None

        for _ in range(network_thinking_time):
            if additive_update:
                network_state = network_state + W.T @ network_state
            else:
                network_state = W.T @ network_state

            if feature_transformation_model is not None:
                network_state = feature_transformation_model(network_state)
            elif activation is not None:
                network_state = activation(network_state)

        return network_state.detach().numpy()


def predict_new_nodes(growth_decision_model, embeddings_for_growth_model, node_embedding_size):
    """
    Predicts which nodes should spawn new nodes.

    Args:
        growth_decision_model: MLP that outputs growth decision
        embeddings_for_growth_model: Node embeddings to evaluate

    Returns:
        Boolean array indicating which nodes spawn new nodes
    """
    new_nodes_predictions = []
    with torch.no_grad():
        predictions_probabilities = growth_decision_model(torch.tensor(embeddings_for_growth_model, dtype=torch.float64)).detach().numpy()
        new_nodes_predictions = (predictions_probabilities > 0).squeeze()

    return new_nodes_predictions



def add_new_nodes(
    G,
    config,
    network_state,
    new_nodes_predictions,
    recorder=None,
    recorder_state=None,
):
    """
    Adds new nodes to the graph based on predictions.
    
    For node-based growth: each node that predicts True spawns a new node
    connected to itself and its neighbors.

    Args:
        G: networkx graph
        network_state: Current node embeddings
        new_nodes_predictions: Boolean array of spawn decisions
        undirected: Whether graph is undirected

    Returns:
        G: Updated graph
        network_state: Updated embeddings including new nodes
    """

    current_graph_size = len(G)

    # Get neighbors for each node BEFORE adding new nodes
    if len(G) == 1:
        neighbors = np.array([[0]])
    else: 
        neighbors = []
        for idx_node in range(len(G)):
            neighbors_idx = [n for n in nx.all_neighbors(G, idx_node)]
            neighbors_idx.append(idx_node)
            neighbors.append(np.unique(neighbors_idx))

    # Add new nodes
    for idx_node in range(len(G)):
        if new_nodes_predictions.shape == ():
            new_nodes_predictions = new_nodes_predictions.reshape(1)

        if new_nodes_predictions[idx_node]:
            if len(neighbors) != 0:
                # Add new node
                G.add_node(current_graph_size)

                # Connect to parent's neighbors
                for neighbor in neighbors[idx_node]:
                    if nx.is_directed(G):
                        G.add_edge(neighbor, current_graph_size, weight=1)
                        G.add_edge(current_graph_size, neighbor, weight=1)
                    else:
                        G.add_edge(neighbor, current_graph_size, weight=1)

                if recorder is not None:
                    recorder(event="node", G=G, new_node=current_graph_size, state=recorder_state)    
                
                current_graph_size += 1

                # Expand network state: new node gets average of parent's neighbors
                new_embedding = np.mean(network_state[neighbors[idx_node]], axis=0, keepdims=True)
                network_state = np.concatenate([network_state, new_embedding], axis=0)

    return G, network_state

def grow_network(evolved_parameters, config, seed=None, verbose=False, record_video=False, video_frames_dir=None, debug=False, debug_label="", debug_cycles=3):    
    """
    Grows a network from initial state using evolved parameters.
    
    This is the main entry point for NDP graph generation.

    Args:
        evolved_parameters: Flattened array of MLP weights and initial embedding
        config: Configuration dictionary
        seed: Random seed

    Returns:
        G: The grown networkx graph
        network_state: Final node embeddings
    """
    
    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    

    frame_idx = 0
    rec_state = {"pos_cache": None}

    def recorder(event, G, state, new_node=None, new_edge=None):
        nonlocal frame_idx
        if video_frames_dir is None:
            return

        if event == "node":
            title = f"Growth: added node {new_node} (N={G.number_of_nodes()} E={G.number_of_edges()})"
            png = os.path.join(video_frames_dir, f"frame_{frame_idx:05d}.png")
            state["pos_cache"] = _render_frame(G, config, png, title=title, highlight_node=new_node, pos_cache=state["pos_cache"])
            frame_idx += 1

        elif event == "edge":
            u, v = new_edge
            title = f"Growth: added edge {u}->{v} (N={G.number_of_nodes()} E={G.number_of_edges()})"
            png = os.path.join(video_frames_dir, f"frame_{frame_idx:05d}.png")
            state["pos_cache"] = _render_frame(G, config, png, title=title, highlight_edge=(u, v), pos_cache=state["pos_cache"])
            frame_idx += 1

    # Create initial graph
    G = generate_initial_graph(
        network_size=config['initial_network_size'],
        sparsity=config['initial_sparsity'],
        undirected=config['undirected'],
        seed=seed
    )
    
    # Build MLPs
    mlp_growth_model = MLP(
        input_dim=config['node_embedding_size'],
        output_dim=1,
        hidden_layers_dims=config['mlp_growth_hidden_layers_dims'],
        last_layer_activated=config['growth_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['growth_model_bias']
    )
    
    mlp_feature_transformation = None
    if config['NN_transform_node_embedding_during_growth']:
        mlp_feature_transformation = MLP(
            input_dim=config['node_embedding_size'],
            output_dim=config['node_embedding_size'],
            hidden_layers_dims=config['mlp_embedding_transform_hidden_layers_dims'],
            last_layer_activated=config['transform_model_last_layer_activated'],
            activation=torch.nn.Tanh(),
            bias=config['transform_model_bias']
        )

    edge_input_size = 2 * config['node_embedding_size']
    mlp_edge = MLP(
        input_dim=edge_input_size,
        output_dim=1,
        hidden_layers_dims=config['mlp_edge_hidden_layers_dims'],
        last_layer_activated=config['edge_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['edge_model_bias']
    )

    # Compute parameter indices for unpacking
    n1 = int(config['nb_params_coevolve_initial_embeddings'] if config['coevolve_initial_embeddings'] else 0)
    n2 = n1 + config['nb_params_growth_model']
    n3 = n2 + config['nb_params_feature_transformation']
    n4 = n3 + config['nb_params_edge_model']



    # Changable initial_network_size (not only 1)
    initN = int(config['initial_network_size'])
    D = int(config['node_embedding_size'])

    if config['coevolve_initial_embeddings']:
        # One shared embedding vector for all nodes
        initial_embedding = np.asarray(evolved_parameters[:n1], dtype=np.float64)
        network_state = np.tile(initial_embedding, (initN, 1))
        # Debugging
        if debug:
            print(f"Network state shape before being passed to prop_feat: {network_state.shape}")

    else:
        # No coevolved embeddings -> random init
        network_state = rng.normal(size=(initN, D))










    # Load evolved weights into growth model
    torch.nn.utils.vector_to_parameters(
        torch.tensor(evolved_parameters[n1:n2], dtype=torch.float64, requires_grad=False),
        mlp_growth_model.parameters()
    )
    
    # Load evolved weights into transform model
    if config['NN_transform_node_embedding_during_growth']:
        torch.nn.utils.vector_to_parameters(
            torch.tensor(evolved_parameters[n2:n3], dtype=torch.float64, requires_grad=False),
            mlp_feature_transformation.parameters()
        )

    # Load evolved weights into edge model
    torch.nn.utils.vector_to_parameters(
        torch.tensor(evolved_parameters[n3:n4], dtype=torch.float64, requires_grad=False),
        mlp_edge.parameters()
    )


    timing = bool(config["timing"])
    timing_verbose = bool(config["timing_verbose"])
    t_grow_total0 = time.perf_counter()

    # Growth loop
    for growth_cycle in range(config['number_of_growth_cycles']):
        # Compute thinking time based on current diameter
        try:
            if config['undirected']:
                diameter = nx.diameter(G)
            else:
                diameter = nx.diameter(G.to_undirected())
        except:
            diameter = max(1, int(np.sqrt(len(G))))
        
        network_thinking_time = diameter + config['network_thinking_time_extra_growth']
        
        if debug and growth_cycle < debug_cycles:
            print(f"{debug_label} Cycle {growth_cycle:03d} Network state shape before MP: {network_state.shape}")
            print(f"{debug_label} Cycle {growth_cycle:03d} network_state before MP: \n {network_state}")
            n_i = network_state.shape[0]
            for node in range(n_i):
                print(f"\nnode {node}:")
                print(f"  in_edges:  {list(G.in_edges(node))}")
                print(f"  out_edges: {list(G.out_edges(node))}")



        # Propagate features
        network_state = propagate_features(
            network_state=network_state,
            W=nx.to_numpy_array(G),
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config['recurrent_activation_function'],
            additive_update=config['additive_update'],
            feature_transformation_model=mlp_feature_transformation
        )

        if debug and growth_cycle < debug_cycles:
            print(f"{debug_label} Cycle {growth_cycle:03d} Network state shape after MP: {network_state.shape}")
            print(f"{debug_label} Cycle {growth_cycle:03d} network_state after MP: \n {network_state}")
            n_i = network_state.shape[0]
            for node in range(n_i):
                print(f"\nnode {node}:")
                print(f"  in_edges:  {list(G.in_edges(node))}")
                print(f"  out_edges: {list(G.out_edges(node))}")
            print("================================")

        # Predict new nodes
        new_nodes_predictions = predict_new_nodes(mlp_growth_model, network_state, config["node_embedding_size"])

        # Add new nodes
        G, network_state = add_new_nodes(
            G=G,
            config=config,
            network_state=network_state,
            new_nodes_predictions=new_nodes_predictions,
            recorder=(recorder if record_video else None), 
            recorder_state=rec_state,
        )

    # Force a final snapshot at the end of grow_network when recording since it records frames when a node (or edge) event happens and not the final state
    if record_video and video_frames_dir is not None:
        title = f"Growth: final (N={G.number_of_nodes()} E={G.number_of_edges()})"
        png = os.path.join(video_frames_dir, f"frame_{frame_idx:05d}.png")
        rec_state["pos_cache"] = _render_frame(G, config, png, title=title, pos_cache=rec_state["pos_cache"])
        frame_idx += 1

    if verbose and timing:
        t_grow_total = time.perf_counter() - t_grow_total0
        print(f"[timing][grow] total={t_grow_total:.4f}s final(N={G.number_of_nodes()} E={G.number_of_edges()})")

    return G, network_state


def count_ndp_parameters(config):
    """
    Counts the number of trainable parameters for NDP.

    Args:
        config: Configuration dictionary

    Returns:
        total_params: Total number of evolved parameters
        Also modifies config in place to store component counts
    """

    # Initial embedding (if co-evolved)
    if config['coevolve_initial_embeddings']:
        D = int(config['node_embedding_size'])
        nb_params_embedding = D

    else:
        nb_params_embedding = 0

    # Growth model parameters
    mlp_growth = MLP(
        input_dim=config['node_embedding_size'],
        output_dim=1,
        hidden_layers_dims=config['mlp_growth_hidden_layers_dims'],
        last_layer_activated=config['growth_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['growth_model_bias']
    )
    nb_params_growth = torch.nn.utils.parameters_to_vector(mlp_growth.parameters()).shape[0]
    
    # Feature transformation parameters
    nb_params_transform = 0
    if config['NN_transform_node_embedding_during_growth']:
        mlp_transform = MLP(
            input_dim=config['node_embedding_size'],
            output_dim=config['node_embedding_size'],
            hidden_layers_dims=config['mlp_embedding_transform_hidden_layers_dims'],
            last_layer_activated=config['transform_model_last_layer_activated'],
            activation=torch.nn.Tanh(),
            bias=config['transform_model_bias']
        )
        nb_params_transform = torch.nn.utils.parameters_to_vector(mlp_transform.parameters()).shape[0]

    edge_input_size = 2 * config['node_embedding_size']
    mlp_edge = MLP(
        input_dim=edge_input_size,
        output_dim=1,
        hidden_layers_dims=config['mlp_edge_hidden_layers_dims'],
        last_layer_activated=config['edge_model_last_layer_activated'],
        activation=torch.nn.Tanh(),
        bias=config['edge_model_bias']
    )
    nb_params_edge = torch.nn.utils.parameters_to_vector(mlp_edge.parameters()).shape[0]

    # Store in config
    config['nb_params_coevolve_initial_embeddings'] = int(nb_params_embedding)
    config['nb_params_growth_model'] = int(nb_params_growth)
    config['nb_params_feature_transformation'] = int(nb_params_transform)
    config['nb_params_edge_model'] = int(nb_params_edge)

    config['nb_trainable_parameters'] = (
        nb_params_embedding
        + nb_params_growth
        + nb_params_transform
        + nb_params_edge
    )

    return config['nb_trainable_parameters']
