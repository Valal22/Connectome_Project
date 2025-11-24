import numpy as np
import networkx as nx


# Convert adjacency matrix to a NetworkX directed graph
def adjacency_to_digraph(A, nodes):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    N = A.shape[0]
    for i in range(N):
        for j in range(N):
            if A[i, j]:
                G.add_edge(nodes[i], nodes[j])
    return G

    
###############
# Eval Metrics
###############

class eval_metrics:
    def __init__(self,
                 in_degrees,
                 out_degrees,
                 hub_neurons,
                 #avg_clustering,
                 avg_shortest_path_length,
                 #small_world,
                 ):

        # Local
        self.in_degrees = in_degrees
        self.out_degrees = out_degrees
        self.hub_neurons = hub_neurons

        # Global
        #self.avg_clustering = avg_clustering
        self.avg_shortest_path_length = avg_shortest_path_length
        # self.small_world = small_world



def compute_eval_metrics(G, verbose=False):
    # 1) Restrict to largest strongly connected component (directed) (as they did in Varshney et al. 2011)
    if nx.is_strongly_connected(G):
        G_lcc = G
    else:
        largest_scc = max(nx.strongly_connected_components(G), key=len)
        G_lcc = G.subgraph(largest_scc).copy() 

    if verbose:
        print(f"G_lcc node count: {len(G_lcc)}")
        print(f"G_lcc type: {type(G_lcc)}")
        print(f"G_lcc edge count: {G_lcc.number_of_edges()}")

    nodes = list(G_lcc.nodes())

    # 2) Local degrees on the directed lcc
    in_degrees_dict = dict(G_lcc.in_degree())
    out_degrees_dict = dict(G_lcc.out_degree())

    in_degrees = np.array([in_degrees_dict[n] for n in nodes], dtype=float)
    out_degrees = np.array([out_degrees_dict[n] for n in nodes], dtype=float)
    total_degrees = in_degrees + out_degrees

    # 3) "Rich-club" (hubs): top 10% by total degree, within the lcc        TO CHECK: IT'S A RAW IMPLEMENTATION AND IT'S NOT MATCHING THE VALUES IN TOWLSON PAPER
    degree_threshold = np.percentile(total_degrees, 90)
    hub_neurons = [n for n, d in zip(nodes, total_degrees) if d >= degree_threshold]

    # 4) Clustering on the DIRECTED, lcc      TO IMPLEMENT
    #avg_clustering = nx.average_clustering(G_lcc, nodes=None, weight=None, count_zeros=True) # This is the standard NetworkX implementation and does not work on Directed graphs  


    # 5) Shortest paths of the DIRECTED, lcc        TO CHECK: IT SEEMS TO MATCH VALUE IN VARSHNEY BUT NEEDS CHECKING
    avg_shortest_path_length = nx.average_shortest_path_length(G_lcc)


    
    # 6) Small-world on lcc     TO IMPLEMENT (high clustering and short path)
    #sigma(G_lcc, niter=100, nrand=10, seed=None)   # Returns the small-world coefficient (sigma) of the G but it does not work for dir G 


    return eval_metrics(
        in_degrees=in_degrees,
        out_degrees=out_degrees,
        hub_neurons=hub_neurons,
        #avg_clustering=avg_clustering,
        avg_shortest_path_length=avg_shortest_path_length,
        #small_world=small_world,
    )