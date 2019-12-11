# Personalized Graph Embedding
A repo for graph embedding with node-personalized attention implemented by Pytorch, based on Attention Walk proposed in https://papers.nips.cc/paper/8131-watch-your-step-learning-node-embeddings-via-graph-attention

### Examples
The following command learns a graph embedding and writes the embedding to disk. The node representations are ordered by the ID.

Creating an Attention Walk embedding of the default dataset wiki-vote with the standard hyperparameter settings.

```
python main.py
```

Creating an Attention Walk embedding of the ppi dataset with the standard hyperparameter settings using global vector attention.

```
python main.py --dataset=ppi --attention=global_vector
```


Creating an Attention Walk embedding of the default dataset with 128 dimensions and learning rate 0.1 using GPU 0 using personalized vector attention.


```
python main.py --emb_dim=128 lr=1e-1 --gpu=0 --attention=personalized_vector

```

