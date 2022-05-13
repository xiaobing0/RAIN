# RAIN
RAIN is an efficient inference system in graph learning with LSH. 

It contains a two-level clustering scheme to fast cluster similar inference batches, reorder them adjacent, and then reuse the repeated nodes among batches to reduce data loading from main memory to GPU.

RAIN also proposes an adaptive tuning strategy to sample the neighbors of the target nodes. Our sampling strategy can significantly reduce the needed nodes while guaranteeing accuracy (a decrease of less than 0.1).
# Prerequisite
* Python 3
* DGL(V==0.7.2)
* 
