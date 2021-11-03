===================== Folder structure: AL4GNN for Node/link classification ==============

1. src/data/load_graph_train.py : load graph, generate train and val masks based on nodes per class

2. src/data/load_graph_test.py : load graph, generate test masks based on nodes per class

3. src/data/config.py : Working files directory

4. utils.py: helping functions

5. src_bgnn/features/build_features.py: helping function for feature extyraction

6. src_bgnn/visualization/visual.py: helping functions for visualizing features

7. src/models/AL_nodeclass.py : Train node classification model with the specified training node labels

8.  src/models/AL_nodeclass_test.py : Test node classification model with the specified testing node labels

===============================================================================================================