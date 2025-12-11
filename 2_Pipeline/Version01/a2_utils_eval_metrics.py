import numpy as np
import networkx as nx

###############################################
# Convert adjacency matrix to a directed graph
###############################################
def adjacency_to_digraph(A, neurons):
    G = nx.DiGraph()
    G.add_nodes_from(neurons)
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if A[i, j]:
                G.add_edge(neurons[i], neurons[j])
    return G


###############################################
# Clustering coefficient (according to Varshney)
###############################################
def varshney_clustering_coefficient(G, node):
    """
    Clustering coefficient for directed graphs as defined in Varshney et al. 2011, equation (8). They used the clustering of 
    the OUT-CONNECTED neighbors "since it captures signal flow emanating from a given neuron".
    
    C_out = EN_out / (k_out * (k_out - 1))
    
    k_out = out-degree
    EN_out = edges among out-neighbors
    """
    out_neighbors = list(G.successors(node))     
    k_out = len(out_neighbors)  
    
    # If less than 2 out-neighbors, clustering is 0. 
    if k_out < 2:
        return 0.0

    # Count edges between out-neighbors
    subgraph = G.subgraph(out_neighbors)    
    EN_out = subgraph.number_of_edges()     
    return EN_out / (k_out * (k_out - 1))

def avg_clustering_varshney(G):
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return 0.0
    total = sum(varshney_clustering_coefficient(G, n) for n in nodes)
    return total / len(nodes)

###############################################
# Getting the Largest Strongly Connected Component (l_scc)
###############################################
def get_largest_scc(G):
    if nx.is_strongly_connected(G):
        return G.copy()
    largest_scc = max(nx.strongly_connected_components(G), key=len)
    return G.subgraph(largest_scc).copy()


###############################################
# Small-world coefficient (according to Varshney)
###############################################
def compute_sw_coeff(G, n_rand=100, verbose=True): 
    G_L = nx.average_shortest_path_length(G)      
    G_C = avg_clustering_varshney(G)
    
    L_rand_list = []
    C_rand_list = []
    n_swaps = G.number_of_edges() * 10
    
    for i in range(n_rand):     # Montecarlo over n_rand randomized graphs (in the original paper they used 1000 times)
        G_rand = G.copy()
        try:
            # Try different function names depending on NetworkX version but they do the same thing: randomize the edges keeping the degree sequence 
            nx.algorithms.swap.directed_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps*100, seed=10+i)
        except AttributeError:
            # Fall back to double_edge_swap (works on DiGraphs in some versions)
            nx.double_edge_swap(G_rand, nswap=n_swaps, max_tries=n_swaps*100, seed=10+i)
        
        G_rand_scc = get_largest_scc(G_rand)
        if G_rand_scc.number_of_nodes() > 1:
            L_rand_list.append(nx.average_shortest_path_length(G_rand_scc))
            C_rand_list.append(avg_clustering_varshney(G_rand_scc))
    
    G_rand_L = np.mean(L_rand_list)
    G_rand_C = np.mean(C_rand_list)
    
    S = (G_C / G_rand_C) * (G_rand_L / G_L)
    
    if verbose:
        print(f"G_rand_scc node count: {len(G_rand_scc)}")
        print(f"G_rand_scc edge count: {G_rand_scc.number_of_edges()}")
        print(f"G_C: {G_C:.4f}, G_rand_C: {G_rand_C:.4f}, C_ratio: {G_C/G_rand_C:.4f}")
        print(f"G_L: {G_L:.4f}, G_rand_L: {G_rand_L:.4f}, L_ratio: {G_rand_L/G_L:.4f}")
    return S

###############################################
# Computing evaluation metrics
###############################################
def compute_eval_metrics(G, verbose=True):
    G_l_scc = get_largest_scc(G)    # G_l_scc is the largest strongly connected component of whatever graph G

    if verbose:
        print(f"G_l_scc node count: {len(G_l_scc)}")
        print(f"G_l_scc type: {type(G_l_scc)}")
        print(f"G_l_scc edge count: {G_l_scc.number_of_edges()}")

    neurons = list(G_l_scc.nodes())

    # Now also compute local degrees and hubs and then return them and the previous metrics (avg clustering, avg path length, small-world coeff)
    # Local degrees on the directed lcc
    in_degrees = np.array([d for _, d in G_l_scc.in_degree()], dtype=float)
    out_degrees = np.array([d for _, d in G_l_scc.out_degree()], dtype=float)
    total_degrees = in_degrees + out_degrees

    # "Rich-club" (hubs): top 5% by total degree, within the lcc        TO CHECK: IT'S A RAW IMPLEMENTATION AND IT'S MATCHING TOWLSON PAPER BUT STILL TO BE VERIFIED
    degree_threshold = np.percentile(total_degrees, 95)
    n_hubs = sum(1 for d in total_degrees if d >= degree_threshold)
    hub_neurons = [n for n, d in zip(neurons, total_degrees) if d >= degree_threshold]
    if verbose:
        print(f"Degree threshold for hubs (95th percentile): {degree_threshold}")
        print(f"List of hub neurons: {hub_neurons}")


    return {
        'nodes': G_l_scc.number_of_nodes(),
        'edges': G_l_scc.number_of_edges(),
        'avg_clustering': avg_clustering_varshney(G_l_scc),
        'avg_path_length': nx.average_shortest_path_length(G_l_scc),
        'small_world_coeff': compute_sw_coeff(G_l_scc),
        'n_hubs': n_hubs,
    }