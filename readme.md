# Active learning for Graph neural networks (AL4GNN)
A scalable active learning framework for GNN tasks on Large graphs.

## Folder structure

### data
processed graphs: edge weights-influence probability, node features, node lables.

### src
1. src/data/config.py: project directory paths
2. src/data/utils.py: utility functions
3. src/data/make_graph.py: preprocess raw graphs 
4. src/data/make_dglgraph.py: prepare graphs for dgl liblary
5. src/data/load_graph.py: load graph in dgl liblary

### models
1. Trained node classification (candidiate node identiifcation) pytorch models.\

Note: There are other files that are developed on fly but are not needed for the output generation.

## Contributors
Rahul Harsha Cheppally 
Aneesh Duraiswaamy 
Sai Munikoti
