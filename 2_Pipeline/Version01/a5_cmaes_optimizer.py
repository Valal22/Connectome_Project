import numpy as np
import cma
import networkx as nx

from a3_models_rand_generator import generate_sbm_graph
from a4_models_train_metr import compute_train_metrics


############################
# Fitness function 
############################
def fitness_function(params, A_target, config):
    """
    It evaluates how well SBM (with given params) matches target connectome.
    
    Parameters are arrays of two elements: [p_in, p_out] which are the SBM edge probabilities
    A_target: np.ndarray (so the target connectome (adjacency matrix))
    config is the dictionary
    
    It returns a float: Loss value (lower is better) and CMA-ES minimizes it.   
        For the metric itself higher is better unless it's 'hamming_norm' where lower is better because it's already the loss
    """
    p_in = float(np.clip(params[0], 0.001, 0.999))
    p_out = float(np.clip(params[1], 0.001, 0.999))
    N = A_target.shape[0]
    
    seed = config['seed']
    n_samples = config['n_samples']
    metric = config['metric']
    max_edges = config['max_edges']
    
    # We will average the loss over n_samples which are SBM graphs generated with the same (p_in, p_out) but different seeds (since it's random)
    losses = []
    for i in range(n_samples):
        G, _, _ = generate_sbm_graph(N, p_in=p_in, p_out=p_out, seed=seed + i, verbose=False)
        A_model = nx.to_numpy_array(G, dtype=int)
        
        m = compute_train_metrics(A_target, A_model)
        
        score = m[metric]
        loss = score if metric == 'hamming_norm' else 1.0 - score
        
        # (Optional) penalty for too many edges
        if config.get('edge_penalty', False) and max_edges is not None:
            model_edges = A_model.sum()
            if model_edges > max_edges:
                penalty = 0.5 * ((model_edges - max_edges) / max_edges)
                loss += penalty
        
        losses.append(loss)
    
    return np.mean(losses)


############################
# The CMA-ES optimizer and optimization loop 
############################
def run_cmaes(A_target, config, verbose=True):
    """
    Run CMA-ES to optimize SBM parameters [p_in, p_out].
    
    Parameters:
    A_target: array adjacency matrix (N x N, binary)
    
    Returns:
    best_params: array. Best [p_in, p_out] found
    best_score: float. Best metric score (e.g., best F1)
    history: pd.DataFrame. Log of optimization progress
    """
    # Get N from target matrix
    N = A_target.shape[0]

    # Retrive config values (values come from config.yaml)
    sigma = config['sigma']
    generations = config['generations']
    popsize = config['popsize']
    seed = config['seed']
    metric = config['metric']
    bounds_p_in = config['bounds_p_in']
    bounds_p_out = config['bounds_p_out']
    minstd = config['minstd']

    # Let's use an initial guess for [p_in, p_out]
    rng = np.random.RandomState(seed)
    initial_p_in  = config.get('initial_p_in', None)
    initial_p_out = config.get('initial_p_out', None)

    if initial_p_in is None:
        initial_p_in = rng.uniform(bounds_p_in[0], bounds_p_in[1])

    if initial_p_out is None:
        initial_p_out = rng.uniform(bounds_p_out[0], bounds_p_out[1])
    
    x0 = [initial_p_in, initial_p_out]
    
    # CMA-ES options
    cma_options = {
        'bounds': [
            [bounds_p_in[0], bounds_p_out[0]],   # Lower bounds
            [bounds_p_in[1], bounds_p_out[1]]    # Upper bounds
        ],
        'maxiter': generations,
        'popsize': popsize,
        'seed': seed,
        'verbose': -9,
        'minstd': minstd,
    }
    
    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(x0, sigma, cma_options)
    
    if verbose:
        print("=" * 60)
        print("CMA-ES Optimization")
        print("=" * 60)
        print(f"  Parameters: [p_in, p_out]")
        print(f"  Metric: {metric}")
        print(f"  Population size: {popsize}")
        print(f"  Max generations: {generations}")
        print(f"  Initial sigma: {sigma}")
        if minstd is not None:
            print(f"  Min std: {minstd}")
        print(f"  Initial guess: p_in={initial_p_in:.4f}, p_out={initial_p_out:.4f}")
        print("=" * 60)

    if not verbose:
        print(f"  Initial guess: p_in={initial_p_in:.4f}, p_out={initial_p_out:.4f}")

    best_score = 0.0
    best_params = None
    best_generation = 0
    gen1_params = None
    gen1_cand_sol = None
    gen5_params = None
    gen5_cand_sol = None
    history = []
    gen = 0

    # The optimization loop
    while not es.stop():
        # 1. Ask: Sample population from current distribution
        solutions = es.ask()
        
        # 2. Evaluate: Compute fitness for each candidate
        fitnesses = [fitness_function(x, A_target, config) for x in solutions]
        
        # 3. Tell: Update distribution based on fitness values
        es.tell(solutions, fitnesses)
        
        gen += 1

        best_idx = np.argmin(fitnesses)
        gen_best_loss = fitnesses[best_idx]
        gen_best_params = solutions[best_idx]
        gen_best_score = 1.0 - gen_best_loss if config['metric'] != 'hamming_norm' else gen_best_loss

        if metric == 'hamming_norm':
            # Only print stats, no output file  
            G_tmp, _, _ = generate_sbm_graph(N, p_in=gen_best_params[0], p_out=gen_best_params[1], seed=seed, verbose=False)
            A_tmp = nx.to_numpy_array(G_tmp, dtype=int)
            m_tmp = compute_train_metrics(A_target, A_tmp)
            n_edges = int(A_tmp.sum())
            density = n_edges / (N * N)
            
            print(f"Gen {gen:3d} | p_in={gen_best_params[0]:.4f} p_out={gen_best_params[1]:.4f} | "
                  f"loss={gen_best_loss:.5f} | edges={n_edges} density={density:.4f} | "
                  f"f1={m_tmp['f1']:.4f} jacc={m_tmp['edge_jaccard']:.4f}")

        else:
            # Specific generations will be tracked. gen1 in order to see the initial progress, gen5 as an intermediate point and then the best
            if gen == 1:
                gen1_params = np.array(gen_best_params).copy()
                gen1_cand_sol = [np.array(s) for s in solutions]

            # gen5
            if gen == 5:
                gen5_params = np.array(gen_best_params).copy()
                gen5_cand_sol = [np.array(s) for s in solutions]

            # best
            if gen_best_score > best_score:
                best_score = gen_best_score
                best_params = np.array(gen_best_params).copy()
                best_generation = gen

            
            history.append({
                'generation': gen,
                'p_in': gen_best_params[0],
                'p_out': gen_best_params[1],
                'score': gen_best_score,
                'best_score': best_score,
                'sigma': es.sigma,
            })

        

    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_generation': best_generation,
        'gen1_params': gen1_params,
        'gen1_cand_sol': gen1_cand_sol,
        'gen5_params': gen5_params,
        'gen5_cand_sol': gen5_cand_sol,
        'history': history,
        'total_generations': gen,
    }
