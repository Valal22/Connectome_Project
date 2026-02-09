import numpy as np
import cma
import networkx as nx
import psutil
import time
import sys
import wandb

_PRINTED_FIRST5_SUBMATRIX = False

from a3_ndp_generator import grow_network, count_ndp_parameters
from a4_train_metr import compute_train_metrics, print_subgraph


# Global variables for parallel evaluation
_GLOBAL_A_TARGET = None
_GLOBAL_CONFIG = None


def _init_worker(A_target, config):
    # Initialize worker process with shared data.
    global _GLOBAL_A_TARGET, _GLOBAL_CONFIG
    _GLOBAL_A_TARGET = A_target
    _GLOBAL_CONFIG = config


def _evaluate_fitness_worker(args):
    """
    Worker function for parallel fitness evaluation.
    MUST always return a finite float (never None), otherwise CMA-ES crashes.
    """
    import numpy as np
    import traceback

    # params = args
    cand_idx, params = args
    cfg = _GLOBAL_CONFIG

    try:
        out = fitness_function(params, _GLOBAL_A_TARGET, cfg, cand_idx=cand_idx)

        # Guard: CMA-ES cannot handle None / NaN / inf
        if out is None:
            return 1e9
        out = float(out)
        if not np.isfinite(out):
            return 1e9

        return out

    except Exception:
        # Print error from worker (important on Windows spawn)
        traceback.print_exc()
        return 1e9
    


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



def fitness_function(params, A_target, config, cand_idx=None):
    """
    It evaluates how well NDP (with given params) matches target connectome.
    
    Args:
        params: Evolved parameters (MLP weights + initial embedding)
        A_target: np.ndarray (so the target connectome (adjacency matrix))
        config is the dictionary
    
    Returns:
        float: Loss value (lower is better). CMA-ES minimizes this!
    """

    N_target_full = A_target.shape[0]
    K = int(config["eval_first_n"])
    K = max(1, min(K, N_target_full))

    A_target_eval = A_target[:K, :K]
    N_target = K
    target_edges = int(A_target_eval.sum())

    seed = config['seed']
    
    n_samples = int(config['n_samples'])

    metric = config["metric"]

    def node_penalty_smooth(n_nodes, N_target):
        diff_N = float(n_nodes - N_target)
        return (diff_N * diff_N)


    def edge_penalty_smooth(n_edges, target_edges, low_ratio=0.90, high_ratio=1.10):
        if target_edges <= 0:
            return 0.0

        r = n_edges / target_edges
        
        if low_ratio <= r <= high_ratio:
            return 0.0
        
        if r < low_ratio:
            diff_E = low_ratio - r
        else:
            diff_E = r - high_ratio

        return diff_E * diff_E

    losses = []

    for i in range(n_samples):
        sample_seed = seed + i * 1000 if seed is not None else None

        # Grow network
        G, _ = grow_network(params, config, seed=sample_seed, debug=False, debug_label=f"[cand {cand_idx}]")

        A_model = nx.to_numpy_array(G, dtype=int)
        N_model = int(A_model.shape[0])

        min_eval = min(N_model, N_target)
        model_edges = int(A_model[:min_eval, :min_eval].sum())

        # Compute penalties
        use_node_pen = config["node_penalty"]
        use_edge_pen = config["edge_penalty"]

        if config["pen_same_scale"] and use_node_pen:
            node_err = abs(N_model - N_target) / N_target 
            node_pen = min(1.0, node_err)   

        else:
            node_pen_raw = node_penalty_smooth(N_model, N_target) if use_node_pen else 0.0
            w_node = config["w_node"]
            node_pen = w_node * node_pen_raw

        if config["pen_same_scale"] and use_edge_pen:
            low_ratio=0.90
            high_ratio=1.10
            r = model_edges / target_edges

            if low_ratio <= r <= high_ratio:
                edge_pen = 0.0
            elif r < low_ratio:
                # how far below the band, relative to the band width
                edge_err = (low_ratio - r) / low_ratio
                edge_pen = min(1.0, edge_err**2)
            else:
                edge_err = (r - high_ratio) / high_ratio
                edge_pen = min(1.0, edge_err**2)
            
        
        else:
            edge_pen_raw = edge_penalty_smooth(model_edges, target_edges) if use_edge_pen else 0.0
            w_edge = config["w_edge"]
            edge_pen = w_edge * edge_pen_raw        
        
        w_wiring = config['w_wiring']

        
        

        # Compute wiring metrics
        min_n = min(N_model, N_target)
        if min_n <= 0:
            losses.append(1.0 + node_pen + edge_pen)
            continue

        A_t = A_target_eval[:min_n, :min_n]
        A_m = A_model[:min_n, :min_n]


        m = compute_train_metrics(A_t, A_m, verbose=False)
        if m is None or metric not in m:
            losses.append(1.0 + node_pen + edge_pen)
            continue

        if metric == 'hamming_norm':
            metric_loss = m[metric]
        else:
            metric_loss = 1.0 - m[metric]

        metric_loss = w_wiring * metric_loss

        if config["all_pen"]:
            loss = metric_loss + node_pen + edge_pen
        else:
            loss = node_pen + edge_pen
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
    
    # Auto-detect target size if not set
    if config['target_network_size'] is None:
        config['target_network_size'] = N
    
    # Count parameters
    nb_parameters = count_ndp_parameters(config)

    # Parameter layout
    n1 = int(config['nb_params_coevolve_initial_embeddings'] if config['coevolve_initial_embeddings'] else 0)
    n2 = n1 + config['nb_params_growth_model']
    n3 = n2 + config['nb_params_feature_transformation']
    n4 = n3 + config['nb_params_edge_model']

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
        print(f"  Samples per eval: {config['n_samples']}")
        print("=" * 60)

    # Physical cores in the machine
    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    use_parallel = num_cores > 1
    
    if use_parallel and sys.platform == 'win32':
        if verbose:
            print(f"  Windows detected - using spawn multiprocessing with {num_cores} workers")

    if use_parallel:
        if verbose:
            print(f"  Using {num_cores} cores (parallel)")
    else:
        if verbose:
            print(f"  Using serial execution")
    
    best_loss = float("inf")

    best_params = None
    best_generation = 0
    gen1_params = None
    gen1_cand_sol = None
    gen5_params = None
    gen5_cand_sol = None
    history = []
    gen = 0


    # Set up pool if using parallel
    pool = None
    if use_parallel:
        from multiprocessing import Pool, get_context
        if sys.platform == 'win32':
            ctx = get_context('spawn')
            pool = ctx.Pool(num_cores, initializer=_init_worker, 
                           initargs=(A_target, config))
        else:
            pool = Pool(num_cores, initializer=_init_worker, 
                       initargs=(A_target, config))


    try:
        # Optimization loop
        while not es.stop() or gen < config["generations"]:

            # Sample population
            solutions = es.ask()

            timing = bool(config["timing"])
            timing_every = int(config["timing_every"])
            t_gen0 = time.perf_counter()


            # Evaluate fitness
            t_eval0 = time.perf_counter()
            if use_parallel and pool is not None:
                fitnesses = pool.map(_evaluate_fitness_worker, [(x) for x in enumerate(solutions)])
            else:
                fitnesses = [fitness_function(x, A_target, config) for x in solutions]
            t_eval = time.perf_counter() - t_eval0

            # Update CMA-ES
            es.tell(solutions, fitnesses)

            t_gen = time.perf_counter() - t_gen0
            if timing and (gen % timing_every == 0):
                pop = len(solutions)
                avg = (t_eval / pop) if pop else float("nan")
                # print(f"[timing][gen {gen}] eval_wall={t_eval:.4f}s gen_wall={t_gen:.4f}s avg={avg:.4f}s pop={pop}")

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
            if verbose and gen % 1 == 0:
                G, _ = grow_network(gen_best_params, config, seed=config["seed"], debug=False)
                n_nodes = G.number_of_nodes()
                n_edges = G.number_of_edges()
                
                # Compute metrics if size matches; otherwise print NA
                f1 = prec = rec = jac = None
                if n_nodes > 0:
                    min_n = min(n_nodes, N)
                    nodes_prefix = list(range(min_n))
                    A_model = nx.to_numpy_array(G, nodelist=nodes_prefix, dtype=int)
                    A_t = A_target[:min_n, :min_n]
                    metr = compute_train_metrics(A_t, A_model, verbose=False)
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
                    f"nodes={n_nodes}/{N} edges={n_edges}/{target_edges}"
                )

                # Neuron name mapping
                if "neurons" in config:
                    neurons = config["neurons"]
                    mapping = []
                    for n in G.nodes():
                        if n < len(neurons):
                            mapping.append(f"{n}:{neurons[n]}")
                        else:
                            mapping.append(f"{n}:UNKNOWN")
                    if config["n_while_generated"]:
                        print("Node mapping (idx:name): " + ", ".join(mapping))


                # ============================================================
                # Plots
                # ============================================================
                if config["plots"] and gen % 500 == 0:        
                    # Create network visualization
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 8))
                    
                    # Spring layout
                    pos = nx.spring_layout(G, seed=0)   # readable 2D layout

                    # Draw network
                    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, width=0.5)
                    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', node_size=500)
                    
                    # Node labels: NeuronName (index)
                    if "neurons" in config:
                        neurons = config["neurons"]
                        labels = {
                            n: f"{neurons[n]} ({n})" if n < len(neurons) else f"UNKNOWN ({n})"
                            for n in G.nodes()
                        }
                    else:
                        labels = {n: str(n) for n in G.nodes()}
                    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=8)
                    
                    ax.set_title(f"Network Graph - Generation {gen}")
                    ax.axis('off')
                    plt.tight_layout()
                    plt.show()




            if config["debug_best_candidate"]:
                # Run once, serially, just for visibility
                _G_dbg, _ = grow_network(
                    gen_best_params,
                    config,
                    seed=config["seed"],
                    debug=True,
                    debug_label=f"[gen {gen} best cand {best_idx}]",
                    debug_cycles=config["debug_cycles"],
                )


            if wandb.run is not None:
                wandb.log({
                    "gen": gen,
                    "best_loss": float(gen_best_loss),
                    "sigma": float(es.sigma),
                    "nodes": int(n_nodes),
                    "edges": int(n_edges),
                    "f1": None if f1 is None else float(f1),
                    "precision": None if prec is None else float(prec),
                    "recall": None if rec is None else float(rec),
                    "edge_jaccard": None if jac is None else float(jac),
                }, step=gen)

        
        if verbose:
            print("=" * 60)
            print(f"Optimization complete")
            print(f"  Best loss: {best_loss:.4f} (gen {best_generation})")
            print(f"  Final sigma: {es.sigma:.6f}")
            print("=" * 60)

    finally:
        if pool is not None:
            pool.close()
            pool.join()
    

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
