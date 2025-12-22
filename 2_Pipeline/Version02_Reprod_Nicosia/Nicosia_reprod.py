import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

inp_dir = Path(__file__).resolve().parent 
out_dir = Path(__file__).resolve().parent / "results" 
out_dir.mkdir(parents=True, exist_ok=True)


def load_data():
    """
    It loads and preprocesses the C. elegans connectome and neuron data about time birth and positions 
    
    Returns:
    - n_t_s: DataFrame with neuron positions and birth t_steps
    - edges: Set of (neuron1, neuron2) tuples representing backbone network (so undirected)
    - adult_degree: Dict mapping neuron name to its degree in adult network
    """
    # Load neuron positions and birth time steps (n_t_s: neuron (n) time (t) and position (s=space))
    n_t_s = pd.read_csv(inp_dir / '2_celegans277_merged.csv')

    # Get valid neurons (those with birth t_steps)
    valid_n = set(n_t_s.loc[n_t_s["birth_time (min)"].notna(), "label"].unique())

    print(f"Valid neurons ({len(valid_n)}) are: {valid_n}")
    print(f"Not valid one is {set(n_t_s['label'].unique()) - valid_n}")

    # Load connectome
    sheet = pd.read_excel(inp_dir / 'Varshney_NeuronConnectFormatted.xlsx')

    # Exclude NMJ (neuromuscular junction) 
    sheet = sheet[sheet['Type'] != 'NMJ']
    

    # Keep only connections where BOTH neurons have birth time data
    sheet_valid = sheet[(sheet['Neuron 1'].isin(valid_n)) & (sheet['Neuron 2'].isin(valid_n))]

    
    # Build the undirected backbone network
    edges = set()
    for _, row in sheet_valid.iterrows():
        n1, n2 = row['Neuron 1'], row['Neuron 2']
        if n1 != n2:  # exclude self-loops
            edge = tuple(sorted([n1, n2]))
            edges.add(edge)
    
    # Compute degree of each neuron in adult network (this is h_j of the model)
    adult_degree = Counter()
    for e in edges:
        adult_degree[e[0]] += 1
        adult_degree[e[1]] += 1
    
    return n_t_s, edges, dict(adult_degree)


def get_t_and_s_points():
    # Helper to get times and lenghts data 

    t_and_s_points = np.array([
        # (time min, length mm)
        (0,    0.050),
        (460, 0.086),       # 1.5 fold (Tapdole) stage
        (770,  0.162),  # within 30 min of hatching (162 um -> 0.162 mm)
        (800,  0.250),   # hatching
        (1080, 0.370),      # L1
        (1530, 0.510),  # L2
        (1860, 0.620),  # L3 
        (2340, 1.050),  # L4
        (2820, 1.130),  # adult
    ], dtype=float)

    return t_and_s_points

def get_interp(t_chosen, t_and_s_points=None):
    if t_and_s_points is None:
        t_and_s_points = get_t_and_s_points()

    # sort by time (important for np.interp)
    t_and_s_points = t_and_s_points[np.argsort(t_and_s_points[:, 0])]
    t_steps = t_and_s_points[:, 0]
    s_coord = t_and_s_points[:, 1]  # s_coord= space (x) coordinates

    return float(np.interp(t_chosen, t_steps, s_coord))

def plot_interp(t_and_s_points):
    t_steps = t_and_s_points[:, 0]
    s_coord = t_and_s_points[:, 1]
    
    # Create smooth interpolation
    t_smooth = np.linspace(0, 2820, 500)
    s_smooth = np.interp(t_smooth, t_steps, s_coord)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Interpolated line
    ax.plot(t_smooth, s_smooth, 'b-', linewidth=2, label='Linear interpolation')
    
    # Known data points
    ax.scatter(t_steps, s_coord, c='red', s=100, zorder=5, label='Known data points')
    
    # Add labels for each stage
    stages = ['Fertilization', 'Tadpole', 'Pre-hatch', 'HATCHING', 'L1', 'L2', 'L3', 'L4', 'Adult']
    for i, (t, l, stage) in enumerate(zip(t_steps, s_coord, stages)):
        offset = 0.05 if i % 2 == 0 else -0.08
        ax.annotate(f'{stage}\n({l*1000:.0f}μm)', (t, l), 
                   textcoords="offset points", xytext=(0, 20 if offset > 0 else -30),
                   ha='center', fontsize=8)
    
    # Hatching line
    ax.axvline(x=800, color='red', linestyle='--', alpha=0.5, label='Hatching (800 min)')
    
    ax.set_xlabel('Time after fertilization (minutes)', fontsize=12)
    ax.set_ylabel('Worm body length (mm)', fontsize=12)
    ax.set_title('C. elegans Body Length During Development\n(Linear Interpolation)', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-50, 3000)
    ax.set_ylim(0, 1.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'interp.png', dpi=150, bbox_inches='tight')
    print(f"Saved interpolation plot to {out_dir / 'interp.png'}")
    plt.close()


def get_d_at_t_btw_2_points(pos_i, pos_j, t_chosen, adult_length=1.13):      # get_d_at_t_btw_2_points: compute distance at a specific time between 2 points
    # ESTG distance:
    # - scale only longitudinal axis (x) (do NOT scale y) as they did in the paper (or at least they omitted to say they scaled also in y-axis)
    # - keep head fixed (x>=0) (this is something I added, it seems to match better the paper results)
    
    # Calculate the scaling factor
    scale = get_interp(t_chosen) / adult_length

    xi = pos_i['x'] if pos_i['x'] >= 0 else pos_i['x'] * scale
    xj = pos_j['x'] if pos_j['x'] >= 0 else pos_j['x'] * scale

    yi, yj = pos_i['y'], pos_j['y']

    return np.sqrt((xi - xj)**2 + (yi - yj)**2)     # so the distance (Pitagorean theorem)



def run_estg_model(n_t_s, adult_degree, delta, seed=None):
    """
    This is the ESTG model to generate a synthetic network.
    
    Alg steps:
    1. Sort neurons by birth time
    2. For each newborn neuron i (in order of birth):
        - For each existing neuron j already in the network:
            - Compute connection probability Π(i→j)
            - Randomly create edge with that probability
    
    Parameters:
    - n_t_s: DataFrame with columns ['label', 'x(mm)', 'y(mm)', 'birth_time (min)']
    - adult_degree: Dict mapping neuron name to degree in adult network
    - delta: The δ parameter controlling typical connection distance, so what is considered close vs far. Example:
        The paper found δ ≈ 0.0126 mm (12.6 micrometers). This means: neurons within 13 um have good chance of connecting
    - seed: Random seed for reproducibility
    
    Returns:
    - growth_curve: List of (N, K) tuples showing network size over growth (N=neurons at a specific time step, K= (related) edges at a specific time step)
    - final_edges: Set of edges in the final generated network
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Sort neurons by birth time and reset the indices after sorting 
    n_sorted = n_t_s.sort_values('birth_time (min)').reset_index(drop=True)

    # Get max degree for normalization
    h_max = max(adult_degree.values())         # h=hub
    
    # Storing neuron positions 
    pos = {}
    for _, row in n_sorted.iterrows():
        pos[row['label']] = {'x': row['x(mm)'], 'y': row['y(mm)']}
    
    # Initializing the network
    existing_n = []  # List of neurons already born      
    edges = set()          # Set of connections formed        
    growth_curve = []      # Track (N, K) as network grows         
    
    # Add neurons one by one in order of birth
    for _, row in n_sorted.iterrows():
        new_n = row['label']
        birth_t = row['birth_time (min)']
        
        # Connect to each existing neuron     
        for n in existing_n:
            h_of_n = adult_degree.get(n, 0)   # h=hub/hubness   
            
            if h_of_n == 0:
                continue
            
            # Compute distance at the time of birth of new neuron. i=new_n and j=n
            d_ij = get_d_at_t_btw_2_points(
                pos[new_n], 
                pos[n], 
                birth_t
            )

            # ESTG connection probability (Key eq (6) in paper)
            # \Pi = (h_j / h_max) * exp(-d_ij(t) / δ)
            prob = (h_of_n / h_max) * np.exp(-d_ij / delta)

            # Bernoulli trial: create edge with probability prob
            if np.random.random() < prob:
                edge = tuple(sorted([new_n, n]))
                edges.add(edge)

        # Add new neuron to the network
        existing_n.append(new_n)
        
        # Record growth state
        N = len(existing_n)
        K = len(edges)
        growth_curve.append((N, K))
        # So, after adding each neuron, we're recording:
            # N = total neurons so far
            # K = total connections so far
        
        # This gives us the growth curve to compare with real data         
    
    return growth_curve, edges


def get_real_growth_curve(n_t_s, real_edges):
    # Compute the real growth curve K(N) from the real C. elegans data.

    # Sort by birth time
    n_sorted = n_t_s.sort_values('birth_time (min)').reset_index(drop=True)
    n_order = n_sorted['label'].tolist()    
    
    # Track which neurons exist at each step
    growth_curve = []
    real_current_n = set()
    
    for n in n_order:
        real_current_n.add(n)
        
        # Count edges that exist between current neurons
        K = sum(1 for e in real_edges if e[0] in real_current_n and e[1] in real_current_n)
        N = len(real_current_n)
        growth_curve.append((N, K))     # So each element in growth_curve is a 2-tuple
    
    return growth_curve


def find_optimal_delta(n_t_s, adult_degree, target_K, n_trials=20):       
    # Find the δ parameter that produces networks with approximately target_K edges.
    
    # They use an iterative bisection (Section S3 of supplement) which is a binary searhc method 
    # So to find a number in a sorted list:
        # 1. Look at the middle element
        # 2. If it's the target: done
        # 3. If target is smaller (so we need MORE edges): search the left half (so we need a LARGER δ)
        # 4. If target is larger (so we need LESS edges): search the right half (so we need a SMALLER δ)
        # 5. Repeat

    # The paper reports optimal δ = 0.0126 for ESTG.
    
    
    print(f"Finding optimal δ to produce K ≈ {target_K} edges...")
    
    delta_low = 0.001
    delta_high = 0.2
    
    for iteration in range(20):
        delta_mid = (delta_low + delta_high) / 2        
        
        # Run multiple trials and average
        K_values = []
        for trial in range(n_trials):
            growth_curve, edges = run_estg_model(n_t_s, adult_degree, delta_mid, seed=trial)
            K_values.append(len(edges))
        
        avg_K = np.mean(K_values)
        
        if abs(avg_K - target_K) / target_K < 0.01:  # So within 1%
            print(f"  Found δ = {delta_mid:.3f}, produces avg K = {avg_K:.1f}")
            return delta_mid
        
        if avg_K < target_K:
            delta_low = delta_mid  # we need more edges > we shift/increase the min δ to an higher value        
        else:
            delta_high = delta_mid  # Too many edges, decrease δ        
        
        if iteration % 5 == 0:
            print(f"  Iteration {iteration}: δ = {delta_mid:.3f}, avg K = {avg_K:.1f}")
    
    print(f"  Final δ = {delta_mid:.3f}, produces avg K = {avg_K:.1f}")
    return delta_mid        

# We can comment the previous find_optimal_delta() if we want to use their optimal delta value (and uncomment the following line) BUT it cannot be used because the implementation (and maybe data and interpolation values) is not the same, in fact the results are bad
# optimal_delta = 0.0126 # Paper one

def plot_fig1D(real_growth_curves, model_growth_curves, n_t_s, hatching_t=800):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left plot: Growth curve K vs N
    ax1 = axes[0]
    
    # Plot real data
    N_obs = [g[0] for g in real_growth_curves]
    K_obs = [g[1] for g in real_growth_curves]
    ax1.scatter(N_obs, K_obs, c='gold', s=20, alpha=0.7, label='Real (C. elegans)', zorder=3)
    
    # Plot model average
    N_model = [g[0] for g in model_growth_curves[0]]
    K_model_all = [[g[1] for g in curve] for curve in model_growth_curves]
    K_model_mean = np.mean(K_model_all, axis=0)
    K_model_std = np.std(K_model_all, axis=0)
    
    ax1.plot(N_model, K_model_mean, 'r-', linewidth=2, label='ESTG model', zorder=2)
    ax1.fill_between(N_model, K_model_mean - K_model_std, K_model_mean + K_model_std, 
                     color='red', alpha=0.2, zorder=1)
    
    # Find N at hatching time
    n_sorted = n_t_s.sort_values('birth_time (min)')
    N_at_hatching = sum(n_sorted['birth_time (min)'] < hatching_t)
    ax1.axvline(x=N_at_hatching, color='red', linestyle='--', alpha=0.7, label=f'Hatching (N≈{N_at_hatching})')
    
    # Fit lines for pre/post hatching
    # Pre-hatching: K  N^2. This corresponds to the Accelerated Topological Growth models (Binomial Accelerated Growth (BAG) and the Hidden-variable Accelerated Growth (HAG)) 
    N_pre = np.array([n for n in N_obs if n < N_at_hatching])
    K_pre = np.array([K_obs[i] for i, n in enumerate(N_obs) if n < N_at_hatching])
    if len(N_pre) > 2:
        # Fit K = a*N^2
        coeffs_pre = np.polyfit(N_pre, K_pre, 2)
        N_fit = np.linspace(1, N_at_hatching, 100)
        K_fit_pre = np.polyval(coeffs_pre, N_fit)       
        ax1.plot(N_fit, K_fit_pre, 'b-', linewidth=1.0, alpha=0.9, label='K ~ N^2 (pre-hatch)', zorder=10)


    # Post-hatching: K  N (linear). It correspond to the Barabási and Albert (BA) model (linear preferential attachment model)
    N_post = np.array([n for n in N_obs if n >= N_at_hatching])
    K_post = np.array([K_obs[i] for i, n in enumerate(N_obs) if n >= N_at_hatching])
    if len(N_post) > 2:
        coeffs_post = np.polyfit(N_post, K_post, 1)
        N_fit = np.linspace(N_at_hatching, max(N_obs), 100)
        K_fit_post = np.polyval(coeffs_post, N_fit)
        ax1.plot(N_fit, K_fit_post, 'g--', linewidth=1.0, alpha=0.9, label='K  N (post-hatch)', zorder=10)
    
    ax1.set_xlabel('N (number of neurons)', fontsize=12)
    ax1.set_ylabel('K (number of edges)', fontsize=12)
    ax1.set_title('Growth Curve: Phase Transition at Hatching', fontsize=12)
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Average degree vs N 
    ax2 = axes[1]
    
    # Compute average degree: <k> = 2K/N
    avg_deg_obs = [2*K/N if N > 0 else 0 for N, K in real_growth_curves]   

    avg_deg_model = [2*k/n if n > 0 else 0 for n, k in zip(N_model, K_model_mean)]     
    
    ax2.scatter(N_obs, avg_deg_obs, c='gold', s=20, alpha=0.7, label='Real')
    ax2.plot(N_model, avg_deg_model, 'r-', linewidth=2, label='ESTG model')
    ax2.axvline(x=N_at_hatching, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('N (number of neurons)', fontsize=12)
    ax2.set_ylabel('<k> (average degree)', fontsize=12)
    ax2.set_title('Average Degree Evolution', fontsize=12)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'fig1D.png', dpi=150, bbox_inches='tight')
    print(f"Saved growth curve plot to {out_dir / 'fig1D.png'}")
    plt.close()


def plot_fig1c(n_t_s, real_growth_curves, model_growth_curves, hatching_t=800, out_file="fig1c_like.png"):
    # Replicating Fig 1C
    n_sorted = n_t_s.sort_values('birth_time (min)').reset_index(drop=True)
    t_steps = n_sorted['birth_time (min)'].to_numpy()
    N = np.arange(1, len(t_steps) + 1)

    K_obs = np.array([K for N, K in real_growth_curves], dtype=float)

    # Decide what K(t) to display prominently
    if model_growth_curves is None:
        K_main = K_obs
        K_std = None
        show_obs_ref = False
        main_label = "K"
    else:
        K_all = np.array([[k for _, k in curve] for curve in model_growth_curves], dtype=float)
        K_main = K_all.mean(axis=0)
        K_std = K_all.std(axis=0)
        show_obs_ref = True
        main_label = "K (ESTG mean)"

    fig, axN = plt.subplots(figsize=(7.2, 3.2))
    axK = axN.twinx()

    # N(t)
    axN.plot(t_steps, N, 'k-', lw=2, label="N")

    # K(t)
    if show_obs_ref:
        axK.plot(t_steps, K_obs, color='0.65', lw=1.5, ls=':', label="K (Real)")
    axK.plot(t_steps, K_main, color='tab:blue', lw=2, ls='--', label=main_label)

    if K_std is not None:
        axK.fill_between(t_steps, K_main - K_std, K_main + K_std, color='tab:blue', alpha=0.15, linewidth=0)

    # Hatching line
    axN.axvline(hatching_t, color='red', ls='--', lw=1.5)
    axN.text(hatching_t + 20, axN.get_ylim()[1] * 0.95, "hatching", color='red', fontsize=10, va='top')

    # Labels + styling
    axN.set_xlabel("Time (min)")
    axN.set_ylabel("N")
    axK.set_ylabel("K")
    axK.tick_params(axis='y', colors='tab:blue')
    axK.yaxis.label.set_color('tab:blue')

    # Legend
    h1, l1 = axN.get_legend_handles_labels()
    h2, l2 = axK.get_legend_handles_labels()
    axN.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9, frameon=False)

    fig.tight_layout()
    fig.savefig(out_dir / out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 1C-style plot to {out_dir / out_file}")           


def main():
    print("=" * 60)
    print("ESTG MODEL - Reproducing Nicosia")
    print("=" * 60)
    
    # 1. Load data
    print("\n[1] Loading data...")
    n_t_s, real_edges, adult_degree = load_data()
    
    N_neurons = len(n_t_s)
    K_edges = len(real_edges)
    print(f"    Neurons with birth times: {N_neurons}")
    print(f"    Edges in backbone network: {K_edges}")
    print(f"    Average degree: {2*K_edges/N_neurons:.2f}")
    max_deg = max(adult_degree.values())
    max_neurons = [name for name, deg in adult_degree.items() if deg == max_deg]
    print(f"    Max degree in adult: {max_deg} (neuron: {', '.join(max_neurons)})")
    
    # 2.0 Compute real growth curve
    print("\n[2] Computing real growth curve...")
    real_growth_curves = get_real_growth_curve(n_t_s, real_edges)    
    
    # Find hatching point
    hatching_t = 800  
    n_sorted = n_t_s.sort_values('birth_time (min)')
    N_at_hatching = sum(n_sorted['birth_time (min)'] < hatching_t)
    K_at_hatching = real_growth_curves[N_at_hatching-1][1] if N_at_hatching > 0 else 0
    print(f"    At hatching (t={hatching_t} min): N={N_at_hatching}, K={K_at_hatching}")
    
    # 3.0 Find optimal δ parameter
    print("\n[3] Finding optimal δ parameter...")
    # If I use just the hardocded optimal delta, comment the following line:
    optimal_delta = find_optimal_delta(n_t_s, adult_degree, target_K=K_edges, n_trials=20)
    print(f"    Optimal δ = {optimal_delta:.3f}")
    
    # 3.1 Get interp data  
    t_and_s_points = get_t_and_s_points()    
    
    # 4. Run model 
    print("\n[4] Running ESTG model (50 trials)...")
    model_growth_curves = []
    model_edges_list = []       # Not used, just to store 
    K_values = []
    
    for trial in range(50):     # it should be 500, but even with 50 the result is the same
        growth_curve, edges = run_estg_model(n_t_s, adult_degree, optimal_delta, seed=trial)
        model_growth_curves.append(growth_curve)
        model_edges_list.append(edges)
        K_values.append(len(edges))
        if (trial + 1) % 10 == 0:
            print(f"    Completed {trial + 1}/50 trials")  

    print(f"    Model produces K = {np.mean(K_values):.1f} ± {np.std(K_values):.1f} edges")
    
    # 5. Plot results
    print("\n[5] Generating plots...")

    plot_interp(t_and_s_points)
    
    plot_fig1D(real_growth_curves, model_growth_curves, n_t_s, hatching_t)
    plot_fig1c(n_t_s, real_growth_curves, model_growth_curves, hatching_t=800, out_file="fig1c_like.png")

    # 6. Summary statistics
    print("\n[6] Summary of results:")
    print("=" * 60)
    print(f"    Number of neurons (N): {N_neurons}")
    print(f"    Number of edges (K): {K_edges}")
    print(f"    Optimal δ parameter: {optimal_delta:.3f}")
    print(f"    Model produces: {np.mean(K_values):.1f} ± {np.std(K_values):.1f} edges")
    
    # Compute growth curve fit quality (as in Table S-II)       
    # N_model = [g[0] for g in model_growth_curves[0]]
    
    K_obs = [g[1] for g in real_growth_curves]
      
    K_model_mean = np.mean([[g[1] for g in curve] for curve in model_growth_curves], axis=0)


    xi = np.abs(np.array(K_obs) - K_model_mean) 
        # That line is computing how far the model’s average number of edges is from the real number of edges.
    print(f"\n    Growth curve fit quality:")
    print(f"    u[ξ(N)] = {np.mean(xi):.1f} (paper: 37.3)")
    print(f"    σ[ξ(N)] = {np.std(xi):.1f} (paper: 31.6)")
    print("=" * 60)
    
    return optimal_delta, model_growth_curves, model_edges_list


if __name__ == "__main__":
    delta, curves, edges = main()


# To run it: python Nicosia_reprod.py 