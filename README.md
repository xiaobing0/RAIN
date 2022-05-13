# RAIN
RAIN is an efficient inference system in graph learning with LSH. RAIN can achieve about 2X to 7X acceleration compared with the original solution in DGL. 

It contains a two-level clustering scheme to fast cluster similar inference batches, reorder them adjacent, and then reuse the repeated nodes among batches to reduce data loading from main memory to GPU.

RAIN also proposes an adaptive tuning strategy to sample the neighbors of the target nodes. Our sampling strategy can significantly reduce the needed nodes while guaranteeing accuracy (decrease less than 0.1).
# Requirements
* Python 3
* DGL(V==0.7.2)
* datasketch
* pymetis
* networkx
# Pre-modify
* Use the new "neighbor.py" to replace the old one in DGL.
* Use the new "lsh.py" to replace the old one in datasketch.
# Run
* Run "RAIN.py" to obtain the inference time of our system on the Reddit dataset.

* Run "DGL_inf.py" to obtain the inference time of the original DGL on the Reddit dataset.
