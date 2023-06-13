# Legion is a GPU-centric System for Large-scale GNN Training.
## GPU-centric Execution
Legion use multi-GPU to accelerate end-to-end GNN training procedure:

1, Graph Sampling

2, Feature Extraction

3, Model Training

## Hierarchical Graph Storage

**GPU Memory**, Unified Cache L1, store the hottest graph topology and features

**CPU Memory**, Unified Cache L2, store graph topology and features with middle hotness

**SSDs**, entire graph topology and features. Each SSD store a specific partition of graph data


![legion-overview](https://github.com/RC4ML/Legion/assets/109936863/c06564d5-21ae-47b2-844e-2c29b234b6b2)


## Cite this work
If you use it in your paper, please cite our work

```
@article{sun2023legion,
  title={Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training},
  author={Sun, Jie and Su, Li and Shi, Zuocheng and Shen, Wenting and Wang, Zeke and Wang, Lei and Zhang, Jie and Li, Yong and Yu, Wenyuan and Zhou, Jingren and others},
  journal={arXiv preprint arXiv:2305.16588},
  year={2023}
}
```
