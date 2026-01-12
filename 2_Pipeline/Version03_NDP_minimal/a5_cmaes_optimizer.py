import numpy as np
import cma
import networkx as nx
import psutil

from a3_ndp_generator import grow_network, count_ndp_parameters
from a4_train_metr import compute_train_metrics



def x0_sampling(dist, nb_parameters):
    """
    Samples initial parameters based on a distribution.
    
    Args:
        dist: Distribution string, e.g., 'U[-1,1]'
        nb_parameters: Number of parameters to sample
        
    Returns:
        Initial parameter vector
    """
    

    if dist == "U[0,1]":
        return np.random.rand(nb_parameters)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(nb_parameters) - 1
    elif dist == "N[0,1]":
        return np.random.randn(nb_parameters)
    else:
        raise ValueError("Distribution not available")

def fitness_function(params, A_target, config):
    """
    It evaluates how well NDP (with given params) matches target connectome.
    
    Args:
        params: Evolved parameters (MLP weights + initial embedding)
        A_target: np.ndarray (so the target connectome (adjacency matrix))
        config is the dictionary
    
    Returns:
        float: Loss value (lower is better). CMA-ES minimizes this!
    """
    N_target = A_target.shape[0]
    target_edges = int(A_target.sum())
    
    seed = config['seed']
    n_samples = config.get('n_samples', 1)
    metric = config['metric']
    size_penalty = config['size_mismatch_penalty']

    losses = []
    
    for i in range(n_samples):
        sample_seed = seed + i if seed is not None else None
        
        # Grow network
        G, _ = grow_network(params, config, seed=sample_seed)
        A_model = nx.to_numpy_array(G, dtype=int)
        
        N_model = A_model.shape[0]
        model_edges = int(A_model.sum())
        
        # Check size match
        if N_model != N_target:
            # Penalize size mismatch proportionally
            size_diff = abs(N_model - N_target) / N_target
            loss = size_penalty * (1.0 + size_diff)
            losses.append(loss)
            continue
        
        # Compute metrics (shapes match)
        m = compute_train_metrics(A_target, A_model)
        
        if m is None:
            # Should not happen if shapes match, but handle gracefully
            losses.append(size_penalty)
            continue
        
        loss = m[metric]
        loss = loss if metric == 'hamming_norm' else 1.0 - loss
        
        # Optional: penalty for too many edges
        if config.get('edge_penalty', False):
            max_edges = config.get('max_edges')
            if max_edges is not None and model_edges > max_edges:
                edge_excess = (model_edges - max_edges) / max_edges
                loss += 0.5 * edge_excess
        
        losses.append(loss)
    
    return np.mean(losses)


def run_cmaes(A_target, config, verbose=True):
    """
    Run CMA-ES to optimize NDP parameters.
    
    Args:
        A_target: Target adjacency matrix (N x N, binary)
        config: Configuration dictionary
        verbose: Print progress
        
    Returns:
        dict with:
            - best_params: Best evolved parameters found
            - best_generation: Generation where best was found
            - gen1_params, gen5_params: Parameters at gen 1 and 5
            - gen1_cand_sol, gen5_cand_sol: All solutions at gen 1 and 5
            - history: List of per-generation stats
            - total_generations: Total generations run
    """
    N = A_target.shape[0]
    target_edges = int(A_target.sum())
    
    # Set max_edges if edge_penalty enabled
    if config['edge_penalty']:
        config['max_edges'] = int(target_edges * config['max_edges_multiplier'])
    
    # Auto-detect target size if not set
    if config['target_network_size'] is None:
        config['target_network_size'] = N
    
    # Count parameters
    nb_parameters = count_ndp_parameters(config)
    

    # Sample initial parameters
    np.random.seed(config["seed"])
    x0 = x0_sampling(config["x0_dist"], nb_parameters)
    
    # CMA-ES options
    cma_options = {
        'maxiter': config["generations"],
        'popsize': config["popsize"],
        'seed': config["seed"],
        'verbose': -9,
        'minstd': config["minstd"],
        'CMA_elitist': config['CMA_elitist'],
    }
    
    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, config["sigma"], cma_options)
    
    if verbose:
        print("=" * 60)
        print("CMA-ES NDP Optimization")
        print("=" * 60)
        print(f"  Target: {N} nodes, {target_edges} edges")
        print(f"  Parameters: {nb_parameters}")
        print(f"  Metric: {config['metric']}")
        print(f"  Population size: {config['popsize']}")
        print(f"  Max generations: {config['generations']}")
        print(f"  Initial sigma: {config['sigma']}")
        print(f"  Growth cycles: {config['number_of_growth_cycles']}")
        print("=" * 60)
    

    # Physical cores in the machine
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    # if verbose:
        # print(f"\nUsing {num_cores} cores\n")

    best_loss = float("inf")

    best_params = None
    best_generation = 0
    gen1_params = None
    gen1_cand_sol = None
    gen5_params = None
    gen5_cand_sol = None
    history = []
    gen = 0
    

    # Optimization loop
    while not es.stop() or gen < config["generations"]:
        # Sample population
        solutions = es.ask()
        
        fitnesses = [fitness_function(x, A_target, config) for x in solutions]

        # Update CMA-ES
        es.tell(solutions, fitnesses)

        gen += 1
        
        # Track best in this generation
        best_idx = np.argmin(fitnesses)

        gen_best_params = solutions[best_idx]
        gen_best_loss = fitnesses[best_idx]


        if gen_best_loss < best_loss:
            best_loss = gen_best_loss
            best_params = np.array(gen_best_params).copy()
            best_generation = gen


        # Track gen1 and gen5
        if gen == 1:
            gen1_params = np.array(gen_best_params).copy()
            gen1_cand_sol = [np.array(s) for s in solutions]
        
        if gen == 5:
            gen5_params = np.array(gen_best_params).copy()
            gen5_cand_sol = [np.array(s) for s in solutions]
        
        
        # Log history
        history.append({
            'generation': gen,
            'best_loss': gen_best_loss,
            'sigma': es.sigma,
        })
        
        # Print progress
        if verbose and gen % config['print_every'] == 0:
            # Generate sample graph for stats
            G, _ = grow_network(gen_best_params, config, seed=config["seed"])
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            
            # Compute metrics if size matches; otherwise print NA
            f1 = prec = rec = jac = None
            if n_nodes == N:
                A_model = nx.to_numpy_array(G, dtype=int)
                metr = compute_train_metrics(A_target, A_model, verbose=False)
                if metr is not None:
                    f1 = metr["f1"]
                    prec = metr["precision"]
                    rec = metr["recall"]
                    jac = metr["edge_jaccard"]

            def fmt(x):
                return f"{x:.4f}" if x is not None else "NA"

            print(
                f"Gen {gen:3d} | loss={gen_best_loss:.4f} | "
                f"f1={fmt(f1)} prec={fmt(prec)} rec={fmt(rec)} jacc={fmt(jac)} | "
                f"nodes={n_nodes} edges={n_edges}"
            )
    
    if verbose:
        print("=" * 60)
        print(f"Optimization complete")
        print(f"  Final sigma: {es.sigma:.6f}")
        print("=" * 60)
    
    return {
        'best_params': best_params,
        'best_generation': best_generation,
        'gen1_params': gen1_params,
        'gen1_cand_sol': gen1_cand_sol,
        'gen5_params': gen5_params,
        'gen5_cand_sol': gen5_cand_sol,
        'history': history,
        'total_generations': gen,
    }
