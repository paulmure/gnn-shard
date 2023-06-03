# Methods of GNN Sharding

## Static Sharding Methods

0. Random (baseline)
1. Gurobi-based 
    - Directly apply Gurobi (ideal, infeasible)
    - Random batch, then apply Gurobi to each batch
    - Small batch obtained by random walk, then apply Gurobi to each batch
2. Community detection (existing work, can be improved)
3. After sharding, estimate the network traffic for each unit, and then offload some nodes from units with high network traffic

# Optimizations

1. Replicate supernodes
    - Motivation: Trade computation and memory with network traffic
2. Dynamic adjustment (assign a node to another unit) and replication (compute the most needed node in each unit)

# Other Important Scenarios to Consider

1. Dynamic sharding (new nodes come and old nodes change during inference)

# Why Cannot SpMM Be Applied Directly

Feature aggregation of GNN can be reduced to SpMM (Sparse Matrix-Matrix Multiplication), but it fails to take into account some key characteristics of feature aggregation:
1. Graphs in GNN is extremely sparse.
2. Power-law distribution of node degree.
3. Feature aggregation on the same graph is applied several times.
4. Graphs can be dynamic.