# Legion is a GPU-initiated system for large-scale GNN training.
```
$ git clone https://github.com/RC4ML/Legion.git
```

## 1. Hardware 
### Hardware Used in Our Paper
All platforms are bare-metal machines.
Table 1
| Platform | CPU-Info | #sockets | #NUMA nodes | CPU Memory | PCIe | GPUs | NVLinks |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DGX-V100 | 96*Intel(R) Xeon(R) Platinum 8163 CPU @2.5GHZ | 2 | 1 | 384GB | PCIe 3.0x16, 4*PCIe switches, each connecting 2 GPUs | 8x16GB-V100 | NVLink Bridges, Kc = 2, Kg = 4 |
| Siton | 104*Intel(R) Xeon(R) Gold 5320 CPU @2.2GHZ | 2 | 2 | 1TB | PCIe 4.0x16, 2*PCIe switches, each connecting 4 GPUs | 8x40GB-A100 | NVLink Bridges, Kc = 4, Kg = 2 |
| DGX-A100 | 128*Intel(R) Xeon(R) Platinum 8369B CPU @2.9GHZ | 2 | 1 | 1TB | PCIe 4.0x16, 4*PCIE switches, each connecting 2 GPUs | 8x80GB-A100 | NVSwitch, Kc = 1, Kg = 8 |

Kc means the number of groups in which GPUs connect each other. And Kg means the number of GPUs in each group.


## 2. Software 
Legion's software is light-weighted and portable. Here we list some tested environment.

1. Nvidia Driver Version: 515.43.04

2. CUDA 11.7

3. GCC/G++ 11.4.0

4. OS: Ubuntu(other linux systems are ok)

5. Intel PCM(according to OS version)
```
$ wget https://download.opensuse.org/repositories/home:/opcm/xUbuntu_18.04/amd64/pcm_0-0+651.1_amd64.deb
```
6. pytorch-cu117, torchmetrics
```
$ pip3 install torch-cu1xx
```
7. dgl 1.1.0
```
$ pip3 install  dgl -f https://data.dgl.ai/wheels/cu1xx/repo.html
```
8. MPI-3.1


## 3. Prepare Datasets and Graph Partitioning
Datasets are from OGB (https://ogb.stanford.edu/), Standford-snap (https://snap.stanford.edu/), and Webgraph (https://webgraph.di.unimi.it/).
Here is an example of preparing datasets for Legion.

### Paper100m dataset
Refer to README in dataset directory for more instructions
```
$ bash prepare_datasets.sh
```

### Partition paper100m
gpu_num represents all gpu numbers you want to use, Legion will partition the graph according to underlying NVlink topology
Note that this step would consume a large volume of CPU memory.
```
$ python graph_partitioning.py --dataset_name 'paper100m' --gpu_num 2
```

## 4. Build Legion from Source

```
$ bash build.sh
```

## 4. Run Legion, Start from training Graphsage on paper100m
There are three steps to train a GNN model in Legion. In these steps, you need to change to **root** user for PCM. (2024.3.11, to solve PCM bugs for general platforms, I disable PCM for now)
### Step 1. Open msr by root for PCM
```
$ modprobe msr
```
### Step 2. Start Legion Server

```
$ python legion_server.py --dataset_path 'dataset' --dataset_name paper100m --train_batch_size 8000 --fanout [25,10] --gpu_number 2 --epoch 2 --cache_memory 38000000 
```

### Step 3. Run Legion Training
After Legion outputs "System is ready for serving", then start training by: 
```
$ python training_backend/legion_graphsage.py --class_num 172  --features_num 128 --hidden_dim 256 --hops_num 2 --gpu_number 2 --epoch 2
```
The training backend will output like this:

<img width="464" alt="100a4006d37d398d2db7ece4edaae97" src="https://github.com/RC4ML/Legion/assets/109936863/1ae401ef-297f-4c88-864a-fe7f8496d973">

I will continuously work on this to improve the running process for easier use.

## Cite this work
If you use it in your paper, please cite our work

```
@inproceedings {sun2023legion,
author = {Jie Sun and Li Su and Zuocheng Shi and Wenting Shen and Zeke Wang and Lei Wang and Jie Zhang and Yong Li and Wenyuan Yu and Jingren Zhou and Fei Wu},
title = {Legion: Automatically Pushing the Envelope of Multi-GPU System for Billion-Scale GNN Training},
booktitle = {2023 USENIX Annual Technical Conference (USENIX ATC 23)},
year = {2023},
pages = {165--179}
}
```

## Future Features of Legion
We will open-source SSD support for Legion in the future.

