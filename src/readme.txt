===================== Folder structure: AL4GNN for Node/link classification ==============

1. src/data/load_graph_train.py : load graph, generate train and val masks based on nodes per class

2. src/data/load_graph_test.py : load graph, generate test masks based on nodes per class

3. src/data/config.py : Working files directory

4. utils.py: helping functions

5. src/features/build_features.py: helping function for feature extyraction

6. src/visualization/visual.py: helping functions for visualizing features

7. src/models/ALNodeclass_Graphsage.py : Train and evaluate node classification model with the queried training node labels.


===============================================================================================================