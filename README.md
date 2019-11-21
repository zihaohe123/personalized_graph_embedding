# personalized_graph_embedding
A repo for graph embedding with node-personalized attention implemented by Pytorch, based on Attention Walk proposed in https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention


### Examples
The following command learns a graph embedding and writes the embedding to disk. The node representations are ordered by the ID.

Creating an Attention Walk embedding of the default dataset with the standard hyperparameter settings.

```
python main.py
```

Creating an Attention Walk embedding of the default dataset with 256 dimensions and using GPU 0.


```
python main.py --emb_dim=256 --gpu=0

```