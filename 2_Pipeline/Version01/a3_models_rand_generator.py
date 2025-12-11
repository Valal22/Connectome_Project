import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph


#########################
# Generate directed Erdos-Renyi random graph 
#########################
def generate_er_graph(num_nodes, p, seed=None):     # p=0.0413 is just a good tuning value to generate a similar amount of edges as the target (this holds for 300 nodes)
    G = erdos_renyi_graph(num_nodes, p, seed=seed, directed=True)
    A = nx.to_numpy_array(G, dtype=int)
    return A, G


#########################
# SBM graph generator (directed)
#########################
def generate_sbm_graph(num_nodes, p_in, p_out, seed=None, verbose=False):
    # Split num_nodes into 3 equal blocks, depending on the num_nodes. p_in=0.16 and p_out=0.02 are just good tuning values to generate a similar amount of edges as the target (this holds for 300 nodes)
    num_blocks = 3
    base_size = num_nodes // num_blocks

    remainder = num_nodes % num_blocks


    sizes = [base_size] * num_blocks

    for i in range(remainder):
        sizes[i] += 1  # distribute any leftover nodes in the first 'remainder' blocks

                
    # Build block probability matrix. This builds a 2D list (a list of lists) called p. It will be a 3×3 matrix.
    p = [[(p_in if i == j else p_out) for j in range(num_blocks)] for i in range(num_blocks)]

    # Now let's call the networkX function to generate the SBM graph 
    G = nx.stochastic_block_model(
        sizes=sizes,
        p=p,
        nodelist=None,       # We are not giving specific node labels. So, NetworkX will label nodes as integers (0, 1, 2, ...) automatically.
        seed=seed,
        directed=True,
        selfloops=False,
        sparse=True,
    )
    

    # quick sanity check print
    if verbose:
        print(
            f"SBM generated: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, sizes={sizes}"
        )

    return G, sizes, p
    # G for analysis,
    # sizes to know the block structure,
    # p to know the edge probabilities
