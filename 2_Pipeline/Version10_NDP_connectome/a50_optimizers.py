from pathlib import Path
import yaml
import torch
import networkx as nx
import numpy as np
from sklearn.metrics import (jaccard_score, hamming_loss, precision_score, 
                             recall_score, f1_score, roc_auc_score)
import psutil
import cma
import time
import sys 
import traceback
import wandb

import pickle

from a30_ndp_generator import grow_network, count_ndp_parameters
from a40_train import compute_train_metrics, soft_edge_balanced_loss_binary_target

HERE = Path.cwd()  
DATASETS = HERE / "datasets"
CONFIG = HERE / "config.yaml"
ADJ_MATRIX = DATASETS / "synapse_count_matrices.xlsx"


def load_config(CONFIG):
    with open(CONFIG, 'r') as file:
        config = yaml.safe_load(file)
    return config
config = load_config(CONFIG)

# Global variables for parallel evaluation
_GLOBAL_A_TARGET = None
_GLOBAL_CONFIG = None

# =====================================================================
# Parallel worker initialization
# =====================================================================
def _init_worker(A_target, config):
    """Initialize worker process with shared data."""
    global _GLOBAL_A_TARGET, _GLOBAL_CONFIG
    _GLOBAL_A_TARGET = A_target
    _GLOBAL_CONFIG = config


# =====================================================================
# Parallel fitness evaluation
# =====================================================================
def _evaluate_fitness_worker(args):
    # params, phase = args
    params = args
    cfg = _GLOBAL_CONFIG

    try:
        out = fitness_function(params, _GLOBAL_A_TARGET, cfg)

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
    
# =====================================================================
# Fitness function
# =====================================================================
def fitness_function(params, A_target, config):
    N_target_full = A_target.shape[0]
    K = int(config["eval_first_n"])
    K = max(1, min(K, N_target_full))

    A_target_eval = A_target[:K, :K]
    N_target = K
    target_edges = int(A_target_eval.sum())

    seed = config['seed']
    
    n_samples = int(config['n_samples'])

    metric = config['metric']

    def node_penalty_smooth(n_nodes, N_target):
        if N_target <= 0:
            return 0.0
        node_err = (n_nodes - N_target) / float(N_target)
        return 10000000 * node_err * node_err   # smooth, grows with distance

    def edge_penalty_smooth(n_edges, target_edges, low_ratio=0.95, high_ratio=1.05):
        if target_edges <= 0:
            return 0.0

        r = n_edges / float(target_edges)

        # deviation below/above the acceptable band
        edge_err_less = max(0.0, low_ratio - r)
        edge_err_more = max(0.0, r - high_ratio)
        
        return 10000 * (edge_err_less**2 + edge_err_more**2)

    losses = []

    for i in range(n_samples):
        sample_seed = seed + i * 1000 if seed is not None else None

        G, _, _ = grow_network(params, config, seed=sample_seed, verbose=False)

        n = G.number_of_nodes()
        A_model = nx.to_numpy_array(G, nodelist=list(range(n)), dtype=int)
        N_model = int(A_model.shape[0])

        min_eval = min(N_model, N_target)
        
        # HARD edge count for penalties: always based on realized adjacency
        model_edges_hard = int(A_model[:min_eval, :min_eval].sum())

        # Optional: keep soft edge count only for logging/inspection 
        model_edges_soft = None
        if "_soft_P" in G.graph:
            P_soft = np.asarray(G.graph["_soft_P"], dtype=np.float64)[:min_eval, :min_eval]
            np.fill_diagonal(P_soft, 0.0)
            model_edges_soft = float(P_soft.sum())

        # Use HARD edges for penalties
        model_edges_hard = float(model_edges_hard)
        
        A_t_sub = A_target_eval[:min_eval, :min_eval]
        target_edges_eff = float(A_t_sub.sum())

        # Compute penalties
        use_node_pen = config["node_penalty"]
        use_edge_pen = config["edge_penalty"]


        node_pen_raw = node_penalty_smooth(N_model, N_target) if use_node_pen else 0.0

        # Guard division by zero if target has no edges
        if target_edges <= 0:
            edge_pen_raw = 0.0
        else:
            # edge_pen_raw = edge_penalty_smooth(model_edges, target_edges) if use_edge_pen else 0.0
            edge_pen_raw = edge_penalty_smooth(model_edges_hard, target_edges_eff) if use_edge_pen else 0.0

        node_pen = node_pen_raw
        edge_pen = edge_pen_raw

        # Compute wiring metrics
        min_n = min(N_model, N_target)
        if min_n < 2:
            # With 0 or 1 node there are no possible (non-self) edges, so wiring metrics are undefined.
            losses.append(1.0 + node_pen + edge_pen)
            continue
        
        A_t = A_target_eval[:min_n, :min_n]
        A_m = A_model[:min_n, :min_n]


        
        if metric == "soft_edge_balanced_loss":
            if "_soft_P" not in G.graph:
                raise ValueError("metric requires store_soft_P=True (missing G.graph['_soft_P'])")

            P_soft = np.asarray(G.graph["_soft_P"], dtype=np.float64)[:min_n, :min_n]
            lam_fp = float(config["lam_fp"])
            base_loss = soft_edge_balanced_loss_binary_target(A_t, P_soft, config)

            if A_t.ndim != 2 or A_t.shape[0] != A_t.shape[1]:
                raise ValueError(f"A_t must be 2D square, got {A_t.shape}")
            if P_soft.shape != A_t.shape:
                raise ValueError(f"P_soft shape {P_soft.shape} must match A_t {A_t.shape}")

            mask = ~np.eye(A_t.shape[0], dtype=bool)
            target_edges_here = float(A_t[mask].sum())
          
            pred_edges_here = float(P_soft[mask].sum())
            w_edgecount = float(config["w_edgecount"])

            edge_count_min_frac = float(config["edge_count_min_frac"])  # 0.85
            edge_count_max_frac = float(config["edge_count_max_frac"])  # 1.15

            E_min = edge_count_min_frac * target_edges_here
            E_max = edge_count_max_frac * target_edges_here

            if target_edges_here <= 0.0:
                edge_penalty = w_edgecount * (pred_edges_here ** 2)
            elif pred_edges_here < E_min:
                edge_penalty = w_edgecount * ((E_min - pred_edges_here) / target_edges_here) ** 2
            elif pred_edges_here > E_max:
                edge_penalty = w_edgecount * ((pred_edges_here - E_max) / target_edges_here) ** 2
            else:
                edge_penalty = 0.0

            metric_loss = base_loss + edge_penalty



        else:
            P_soft_local = None
            if "_soft_P" in G.graph:
                P_soft_local = np.asarray(G.graph["_soft_P"], dtype=np.float64)[:min_n, :min_n]
            else:
                raise ValueError("compute_train_metrics requires soft_A_model; set store_soft_P=True")

            m = compute_train_metrics(A_t, A_m, P_soft_local, config=config)
            if m is None or metric not in m:
                losses.append(1.0 + node_pen + edge_pen)
                continue

            if metric == 'hamming_norm':
                metric_loss = float(m[metric])
                    
            else:
                metric_loss = 1.0 - m[metric]

        loss = metric_loss + node_pen + edge_pen
        losses.append(loss)

    return np.mean(losses)


# =====================================================================
# Initial parameter sampling
# =====================================================================
def x0_sampling(dist, nb_parameters):
    if dist == "U[0,1]":
        return np.random.rand(nb_parameters)
    elif dist == "U[-1,1]":
        return 2 * np.random.rand(nb_parameters) - 1
    elif dist == "N[0,1]":
        return np.random.randn(nb_parameters)
    elif dist == "U[0,0]":
        return np.zeros(nb_parameters)
    else:
        raise ValueError("Distribution not available")
    


# =====================================================================
# CMA-ES optimization loop
# =====================================================================
def run_cmaes(A_target, config, verbose=True):
    N = A_target.shape[0]
    target_edges = int(A_target.sum())
    
    if config['target_network_size'] is None:
        config['target_network_size'] = N
    
    nb_parameters = count_ndp_parameters(config)

    np.random.seed(config["seed"])
    x0 = x0_sampling(config["x0_dist"], nb_parameters)


    # Inductive bias: shift growth-MLP last-layer bias positive so spawning is
    # the default at gen 0. CMA-ES is free to evolve away from this if needed.
    n1 = int(config["nb_params_coevolve_initial_embeddings"]) if config["coevolve_initial_embeddings"] else 0
    n2 = n1 + config["nb_params_growth_model"]
    growth_last_bias_idx = n2 - 1
    x0[growth_last_bias_idx] += float(config["growth_bias_init"])


    cma_options = {
        'maxiter': config["generations"],
        'popsize': config["popsize"],
        'seed': config["seed"],
        'verbose': -9,
        'minstd': config["minstd"],
        'CMA_elitist': config['CMA_elitist'],
    }

    if verbose:
        print("=" * 60)
        print("CMA-ES NDP Optimization")
        print("=" * 60)
        print(f"  Target: {N} nodes, {target_edges} edges")
        print(f"  Parameters: {nb_parameters}")
        print(f"  Metric: {config['metric']}")
        print(f"  Population size: {config['popsize']}")
        print(f"  Max generations: {config['generations']}")
        print(f"  Samples per eval: {config.get('n_samples', 1)}")
        print(f"  Use restarts: {config['use_restarts']}")
        if config['use_restarts']:
            print(f"  Restart: warmup={config['restart_warmup']}, "
                  f"patience={config['restart_patience']}, "
                  f"rel_tol={config['restart_rel_tol']}, "
                  f"max={config['restart_max_count']}, "
                  f"from_best={config['restart_from_best']}")
        print("=" * 60)

    num_cores = psutil.cpu_count(logical=False) if config["threads"] == -1 else config["threads"]
    use_parallel = num_cores > 1

    if use_parallel and verbose:
        print(f"  Using {num_cores} cores (parallel)")
    elif verbose:
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
    best_loss_per_gen = []   # best-so-far 

    # Restart state
    restarts_done = 0
    last_restart_gen = 0
    last_check_gen = None             # set when we hit warmup
    last_check_best_loss = None       # running best snapshotted at last_check_gen


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
        # Initial ES instance
        current_x0 = x0
        current_sigma = float(config["sigma"])
        es = cma.CMAEvolutionStrategy(current_x0, current_sigma, cma_options)

        # Optimization loop with optional manual restarts
        while gen < config["generations"]:

            # Inner loop: run current ES until its own stop, manual restart, or maxgen
            # while not es.stop() and gen < config["generations"]:
            while gen < config["generations"]:

                solutions = es.ask()

                t_gen0 = time.perf_counter()

                if use_parallel and pool is not None:
                    fitnesses = pool.map(_evaluate_fitness_worker, [(x) for x in solutions])
                else:
                    fitnesses = [fitness_function(x, A_target, config) for x in solutions]

                es.tell(solutions, fitnesses)
                t_gen = time.perf_counter() - t_gen0

                gen += 1

                best_idx = np.argmin(fitnesses)
                gen_best_params = solutions[best_idx]
                gen_best_loss = fitnesses[best_idx]

                if gen_best_loss < best_loss:
                    best_loss = gen_best_loss
                    best_params = np.array(gen_best_params).copy()
                    best_generation = gen

                best_loss_per_gen.append(best_loss)

                if gen == 1:
                    gen1_params = np.array(gen_best_params).copy()
                    gen1_cand_sol = [np.array(s) for s in solutions]

                if gen == 5:
                    gen5_params = np.array(gen_best_params).copy()
                    gen5_cand_sol = [np.array(s) for s in solutions]

                history.append({
                    'generation': gen,
                    'best_loss': gen_best_loss,
                    'sigma': es.sigma,
                    'restart_idx': restarts_done,
                })

                # Print progress
                if verbose and (gen % int(config["print_every"]) == 0):
                    G, _, _ = grow_network(gen_best_params, config, seed=config["seed"], verbose=False)

                    n_nodes = G.number_of_nodes()
                    n_edges = G.number_of_edges()

                    f1 = prec = rec = jac = auroc = soft_edge_ed = None

                    if n_nodes > 1:
                        min_n = min(n_nodes, N)
                        nodes_prefix = list(range(min_n))
                        A_model = nx.to_numpy_array(G, nodelist=nodes_prefix, dtype=int)
                        A_t = A_target[:min_n, :min_n]
                        if "_soft_P" not in G.graph:
                            raise KeyError("G.graph['_soft_P'] missing; set store_soft_P=True in config")
                        P_soft = np.asarray(G.graph["_soft_P"], dtype=np.float64)[:min_n, :min_n]
                        metr = compute_train_metrics(A_t, A_model, P_soft, config=config)
                        if metr is not None:
                            f1 = metr["f1"]
                            prec = metr["precision"]
                            rec = metr["recall"]
                            jac = metr["edge_jaccard"]
                            auroc = metr["auroc"]
                        soft_edge_ed = soft_edge_balanced_loss_binary_target(A_t, P_soft, config)

                    def fmt(x):
                        return f"{float(x):.4f}" if x is not None else "NA"

                    print(
                        f"Gen {gen:3d} | loss={float(gen_best_loss):.4f} | "
                        f"f1={fmt(f1)} prec={fmt(prec)} rec={fmt(rec)} jacc={fmt(jac)} | "
                        f"nodes={n_nodes}/{N} edges={n_edges}/{target_edges} | "
                        f"soft_edge_ed={fmt(soft_edge_ed)} | "
                        f"auroc={fmt(auroc)} | r={restarts_done}",
                        flush=True,
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
                            "soft_edge_ed": None if soft_edge_ed is None else float(soft_edge_ed),
                            "auroc": None if auroc is None else float(auroc),
                            "restart_idx": restarts_done,
                        }, step=gen)

                # Periodic check
                if config["use_restarts"]:
                    warmup = int(config["restart_warmup"])
                    patience = int(config["restart_patience"])

                    # Snapshot anchor exactly at gen == warmup (e.g. gen 400)
                    if last_check_best_loss is None and gen == warmup:
                        last_check_gen = gen
                        last_check_best_loss = best_loss

                    # Fire a check only at gen == last_check_gen + patience (550, 700, 850, ...)
                    if (last_check_best_loss is not None
                            and restarts_done < config["restart_max_count"]
                            and gen == last_check_gen + patience
                            and best_params is not None):

                        recent_best = best_loss
                        past_best = last_check_best_loss
                        improvement = past_best - recent_best
                        rel_improvement = improvement / max(abs(recent_best), 1e-12)

                        if verbose:
                            print(
                                f"[Check] gen {gen} vs gen {last_check_gen}: "
                                f"past={past_best:.6f} recent={recent_best:.6f} "
                                f"rel_improvement={rel_improvement:.6f} "
                                f"threshold={config['restart_rel_tol']}",
                                flush=True,
                            )

                        # Advance anchor BEFORE possibly breaking out
                        last_check_gen = gen
                        last_check_best_loss = best_loss

                        if rel_improvement < float(config["restart_rel_tol"]):
                            if verbose:
                                print(
                                    f"[Restart {restarts_done+1}] triggered at gen {gen}",
                                    flush=True,
                                )
                            break  # rebuild ES

            # End of inner loop. Decide why we exited.
            if gen >= config["generations"]:
                break
            if es.stop() and (not config["use_restarts"] or restarts_done >= config["restart_max_count"]):
                if verbose:
                    print(f"[Stop] es.stop()={es.stop()} at gen {gen}, no more restarts available.", flush=True)
                break


            if gen < int(config["restart_warmup"]):
                if verbose:
                    print(
                        f"[Stop] es.stop()={es.stop()} at gen {gen}, "
                        f"before restart_warmup={config['restart_warmup']}; not restarting.",
                        flush=True,
                    )
                break


            if (gen - last_restart_gen) < int(config["restart_patience"]):
                if verbose:
                    print(
                        f"[Stop] CMA-ES stopped at gen {gen}, but only "
                        f"{gen - last_restart_gen} gens since last restart; "
                        f"restart_patience={config['restart_patience']}; not restarting.",
                        flush=True,
                    )
                break

            # Rebuild ES (either because of es.stop() or stagnation)
            restarts_done += 1
            last_restart_gen = gen

            if config["restart_from_best"]:
                current_x0 = best_params.copy()
            else:
                np.random.seed(config["seed"] + restarts_done)
                current_x0 = x0_sampling(config["x0_dist"], nb_parameters)

            current_sigma = float(config["restart_sigma"])
            # Reuse cma_options but bump the seed so internal RNG differs across restarts
            cma_options_restart = dict(cma_options)
            cma_options_restart["seed"] = int(cma_options["seed"]) + restarts_done

            es = cma.CMAEvolutionStrategy(current_x0, current_sigma, cma_options_restart)

            if verbose:
                print(
                    f"[Restart {restarts_done}/{config['restart_max_count']}] at gen {gen}: "
                    f"x0={'best' if config['restart_from_best'] else 'random'}, sigma={current_sigma}",
                    flush=True,
                )

        if verbose:
            print("=" * 60)
            print(f"Optimization complete")
            print(f"  Best loss: {best_loss:.4f} (gen {best_generation})")
            print(f"  Final sigma: {es.sigma:.6f}")
            print(f"  Restarts used: {restarts_done}")
            print("=" * 60)


    finally:
        if pool is not None:
            pool.close()
            pool.join()
    # NEW 5: Restarting

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
