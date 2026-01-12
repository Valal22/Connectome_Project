import numpy as np
import torch
import networkx as nx
from scipy import sparse
from scipy.stats import uniform
from numpy.random import default_rng

# Disable gradients globally for inference
torch.autograd.set_grad_enabled(False)


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
    return torch.nn.Sequential(*layers)


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
        else:
            G_check = nx.from_numpy_array(W, create_using=nx.DiGraph)
        
        # For directed graphs, check weak connectivity
        if undirected:
            if nx.is_connected(G_check):
                break
        else:
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


def predict_new_nodes(growth_decision_model, embeddings_for_growth_model):
    """
    Predicts which nodes should spawn new nodes.

    Args:
        growth_decision_model: MLP that outputs growth decision
        embeddings_for_growth_model: Node embeddings to evaluate

    Returns:
        Boolean array indicating which nodes spawn new nodes
    """
    with torch.no_grad():
        predictions = growth_decision_model(
            torch.tensor(embeddings_for_growth_model, dtype=torch.float64)
        ).detach().numpy()
        # Positive output = spawn new node
        new_nodes_predictions = (predictions > 0).squeeze()
    
    return new_nodes_predictions


def add_new_nodes(G, network_state, new_nodes_predictions, undirected, config, max_nodes=None):
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
    
    # Handle scalar prediction (single node case)
    if new_nodes_predictions.shape == ():
        new_nodes_predictions = new_nodes_predictions.reshape(1)
    
    # Get neighbors for each node BEFORE adding new nodes
    if len(G) == 1:
        neighbors = [[0]]
    else:
        neighbors = []
        for idx_node in range(len(G)):
            neighbors_idx = list(nx.all_neighbors(G, idx_node))
            neighbors_idx.append(idx_node)  # Include self
            neighbors.append(np.unique(neighbors_idx))
    
    # Add new nodes
    for idx_node in range(min(len(new_nodes_predictions), len(neighbors))):
        if config['max_nodes_cap']:
            if max_nodes is not None and current_graph_size >= max_nodes:
                break
        
        
        if new_nodes_predictions[idx_node]:
            if config['max_nodes_cap']:
                if max_nodes is not None and current_graph_size >= max_nodes:
                    break
            
            # Add new node
            G.add_node(current_graph_size)
            
            # Connect to parent's neighbors
            for neighbor in neighbors[idx_node]:
                if undirected:
                    G.add_edge(neighbor, current_graph_size, weight=1)
                else:
                    # Directed: bidirectional edges
                    G.add_edge(neighbor, current_graph_size, weight=1)
                    # G.add_edge(current_graph_size, neighbor, weight=1)
                    
                    # G.add_edge(idx_node, current_graph_size, weight=1)

            # Expand network state: new node gets average of parent's neighbors
            new_embedding = np.mean(network_state[neighbors[idx_node]], axis=0, keepdims=True)
            network_state = np.concatenate([network_state, new_embedding], axis=0)
            
            current_graph_size += 1
    
    return G, network_state


def grow_network(evolved_parameters, config, seed=None, verbose=True):
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
    max_nodes = config.get("target_network_size", None)

    # Set seed if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create initial graph
    G = generate_initial_graph(
        network_size=config['initial_network_size'],
        sparsity=config['initial_sparsity'],
        undirected=config['undirected'],
        seed=seed
    )
    
    # Build MLPs
    input_size_growth_model = config['node_embedding_size']  # node-based growth
    
    mlp_growth_model = MLP(
        input_dim=input_size_growth_model,
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
    
    # Compute parameter indices for unpacking
    n1 = config['node_embedding_size'] if config['coevolve_initial_embeddings'] else 0
    n2 = n1 + config['nb_params_growth_model']
    n3 = n2 + config['nb_params_feature_transformation']
    





    # Initialize network state
    # Original
    # if config["coevolve_initial_embeddings"]:
    #     network_state = np.expand_dims(evolved_parameters[: config["node_embedding_size"]], axis=0)

    # Changable initial_network_size (not only 1)
    if config['coevolve_initial_embeddings']:
        initial_embedding = evolved_parameters[:n1]
        network_state = np.tile(initial_embedding, (config['initial_network_size'], 1))
    
    else:
        network_state = np.ones((config['initial_network_size'], config['node_embedding_size']))
    





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
        
        # Propagate features
        network_state = propagate_features(
            network_state=network_state,
            W=nx.to_numpy_array(G),
            network_thinking_time=network_thinking_time,
            recurrent_activation_function=config['recurrent_activation_function'],
            additive_update=config['additive_update'],
            # persistent_observation=persistent_observation, present in the source! 
            feature_transformation_model=mlp_feature_transformation
        )
        
        # Predict new nodes (node-based growth)
        new_nodes_predictions = predict_new_nodes(mlp_growth_model, network_state)
        
        # Add new nodes
        G, network_state = add_new_nodes(
            G=G,
            network_state=network_state,
            new_nodes_predictions=new_nodes_predictions,
            undirected=config['undirected'],
            config=config,
            max_nodes=max_nodes,
        )
        
        if verbose:
            print(f"Grown graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
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
    nb_params_embedding = config['node_embedding_size'] if config['coevolve_initial_embeddings'] else 0
    
    # Growth model parameters
    input_size = config['node_embedding_size']  # node-based growth
    mlp_growth = MLP(
        input_dim=input_size,
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
    
    # Store in config
    config['nb_params_coevolve_initial_embeddings'] = nb_params_embedding
    config['nb_params_growth_model'] = nb_params_growth
    config['nb_params_feature_transformation'] = nb_params_transform
    config['nb_trainable_parameters'] = nb_params_embedding + nb_params_growth + nb_params_transform
    
    return config['nb_trainable_parameters']
