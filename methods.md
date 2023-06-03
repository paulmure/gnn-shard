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
2. Dynamic adjustment (assign a node to another unit) and replication (compute the most needed node in each unit)

# Other Important Scenarios to Consider

1. Dynamic sharding (new nodes come and old nodes change during inference)
