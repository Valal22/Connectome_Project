import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph


#########################
# Random graph generator
#########################

def generate_er_random_graph(num_nodes=300, p=0.0413, seed=12):
    G_rand = erdos_renyi_graph(num_nodes, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G_rand, dtype=int)
    return A, G_rand