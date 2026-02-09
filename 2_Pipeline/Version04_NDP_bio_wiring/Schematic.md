# Inut/output
**Input:**
- Dataset: connectome with connections, birth times, 2Dcoordinates
- Hyperparameters: embedding dimension, growth cycles $C$, generations $T$, population size $\lambda$, metrics, penalties, bias

**Output:**
- Optimized parameters $\theta^*$ ($^*$ means optimized/best)
    - **Parameter layout:** $\theta = [E_{\text{init}} \,|\, E_{\text{io}} \,|\, W_{\mathcal{R}} \,|\, W_{\mathcal{T}} \,|\, W_{\mathcal{W}}]$
    - $E_{\text{io}}$ (embeddings input/output is work in progress)
- Generated graph $G^*$
- Metrics (F1, Jaccard, precision, recall and -optionally- small-world coefficient, average clustering, average shortest path length)

---

# Code schematic
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DATA LOADING AND INITIALIZATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
- Load target adjacency matrix A* and neuron list from dataset
- Load neuron birth order, sort by birth_time ascending
- Load neuron coordinates in mm for each neuron
- Optionally truncate to the first K neurons (eval_first_n) for speed
- Initialize node embeddings and three MLPs deciding for (1) growth ($R$), (2) emb transformation ($MP$), (3) edge establishment ($W$)
- Set up four wiring biases (applied in configurable order):
    - Locality: Implement distance calculation between nodes (closer > higher likelyhood to connect)
    - Class: neuron generated will belong to one of these classes: sensory, interneuron, motor. This affects how connects to who 
    - Rich-club: hub neurons receive bonus for connections
    - Work in progress: send/receive I/O embeddings, for each generated node, to be learned
- Count total number of evolvable parameters p

- Initialize CMA-ES:
    - Sample initial parameter vector $x_0$ $\in$ $\mathbb{R} ^p$ from x0_dist ("U[-1,1]" or "U[0,1]" or "N[0,1]")
    - Initialize CMA-ES with ($x_0$, sigma $\sigma$, population size $\lambda$)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CMA-ES EVOLUTION LOOP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
- for gen in Generations do
    - Sample $\lambda$ candidate parameter vectors ($\theta_i$) from CMA-ES
    - for each candidate $\theta_i$ do (**NDP FORWARD PASS: GROW A NETWORK FROM** $\theta_i$)
        - Create a small random initial graph G (Default: 1 node)
        - Initialize node embeddings $E$ from $\theta_i$ (Default: co-evolved -or random-)
        - Unpack $\theta_i$ into MLP weights for $R$, $MP$, $W$ (Replication, Transform and wiring models respectively)

        - for growth_cycle in C do

            - // Message passing
                - Compute thinking time $\tau$ = diameter + extra
                - Repeat $\tau$ times: $E \leftarrow W^T _{adj} \times E$ (E=network_state), then apply $MP(E)$ 
                                (Note: transpose means info flows from predecessors to successors or: it computes the sum of messages received by the current considered node)

            - // Node replication
                - For each existing node i: if $R(E_i) > 0$, spawn a new node
                - New node inherits embedding = mean of parent's neighborhood

            - // Wiring 
                - Remove ALL edges from G (it avoids summing of edges across cycles)
                - For every ordered pair (i, j), $i \ne j$ (avoiding self-loops for now):
                    - Accumulate logit $l$ = 0, applying biases in configured order:
                        - "mlp": $l$ += W(input) (input: concatenated network_state of i and j) AND/OR
                        - "locality":  $l$ += $− \alpha \cdot (\frac{d_{ij}}{\sigma})^2$ AND/OR
                        - "class": $l$ += class_logit $[c_i, c_j]$ + pre_bias $[c_i]$ + post_bias $[c_j]$ AND/OR
                        - "rich_club": $l$ += bonuses if i and/or j are hub neurons
                    - $P_{ij}$: sigmoid($l$), so we turn the number in a probability 
                - Establish edges by Bernoulli: sample each edge independently with probability $P_{ij}$: toss a rand number and if $< P_{ij}$ establish connection 
        - end for (growth cycles)

    - **LOSS COMPUTATION**            
        - Convert G to adjacency matrix A_model
        - Align A_model and A* to their common first min(N_model, N_target) rows/cols
        - Compute chosen wiring metric (F1, or Jaccard etc)
        - L_metric: 1 − metric_value  (lower is better)
        - L_node: $node_{err} = \frac{(n_{nodes} - N_{target})}{float(N_{target})}$ $= node_{err}^2 \cdot 10000$ 
        - L_edge: $edge_{err} = \frac{(n_{edges} - target_{edges})}{float(target_{edges})}$ $= edge_{err}^2 \cdot 1000$ 
        - $f_i =  w_{wiring} \cdot L_{metric} + w_{node} \cdot L_{node} + w_{edge} \cdot L_{edge}$ 
            - $f_i$: a float 
            - $w_{wiring}, w_{node} \space \text{and} \space w_{edge}$: weight for each loss 

        - end for (population)

    - CMA-ES update: tell({$\theta _i$, $f_i$}) $\rightarrow$ update mean and covariance
    - if min($f_i$) < $f^*$ then update ($\theta^*$, $f^*$) ($^*$ means best here)

    - end for (generations)
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL RECONSTRUCTION AND REPORTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
- Re-run NDP forward pass with $\theta ^*$ $\rightarrow$ G*
- Compute training metrics: F1, Jaccard, Precision, Recall
- Optionally compute eval metrics on largest SCC (strong connected component) of G*: average clustering coefficient, average shortest path length, small-world coefficient 
- Save $\theta ^*$, adjacency matrix, metrics, growth video
- return $\theta ^*$, G*, metrics

---

# Notation Reference

| Symbol | Meaning |
|--------|---------|
| $A^*$ | Target adjacency matrix (binary, $N \times N$) (so the connectome)|
| $\theta$ | Flattened parameter vector |
| $E$ | Node embedding matrix ($n \times D$) |
| $\mathcal{R}$ | Replication MLP (growth decision) |
| $\mathcal{T}$ | Transform MLP (embedding update) |
| $\mathcal{W}$ | Wiring MLP (edge probability) |
| $W_{\text{adj}}$ | Adjacency matrix |
| $\alpha, \sigma$ | Locality bias strength ($\alpha$) and length scale ($\sigma$)|
| $L_{\text{class}}$ | Class-to-class logit matrix |

---
