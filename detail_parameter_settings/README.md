# Legion-ATC23-Artifacts
This file might helps you to reproduce the main results.
I'm continusly working on this projects to ease the use.

## Datasets
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- | --- | --- |
| #Vertices | 2.4M | 111M | 65M | 133M | 0.79B | 1B |
| #Edges | 120M | 1.6B | 1.8B | 5.5B | 47.2B | 42.5B |
| Feature Size | 100 | 128 | 256 | 256 | 128 | 128 |
| Topology Storage | 640MB | 6.4GB | 7.2GB | 22GB | 189GB | 170GB |
| Feature Storage | 960MB | 56GB | 65GB | 136GB | 400GB | 512GB |
| Class Number | 47 | 2 | 2 | 2 | 2 | 2 |

## Hyper-parameter settings
To reproduce the results in paper, we need a 8-GPU machine and run the Python scripts, legion_server.py legion_graphsage.py for graphsage model. (legion_gcn.py for gcn model). The hyper-parameters in Python scripts are shown below.
### Figure 8 DGX-V100, Legion hyper-parameters:
| Datasets | PR | PA | CO | UKS |
| --- | --- | --- | --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 | 
| cache_memory | 13GB | 13GB | 11GB | 11GB |
| usenvlink | 1 | 1 | 1 | 1 | 
| class_num | 47 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 
| hidden_dim | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 

### Figure 8 DGX-A100, Legion hyper-parameters:
| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- |  --- | --- | 
| train_batch_size | 8000 | 8000 | 8000 | 8000 | 8000 | 8000 |
| epoch | 10 | 10 | 10 | 10 | 10 | 10 | 
| gpu_number | 8 | 8 | 8 | 8 |  8 | 8 | 
| cache_memory | 36GB | 36GB | 32GB | 32GB | 36GB | 36GB |
| usenvlink | 1 | 1 | 1 | 1 | 1 | 1 | 
| class_num | 47 | 2 | 2 | 2 | 2 | 2 | 
| features_num | 100 | 128 | 256 | 256 | 128 | 128 |
| hidden_dim | 256 | 256 | 256 | 256 | 256 | 256 |
| drop_rate | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 
| learning_rate | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 


All systems will output the epoch time of each setting. 

If SEGMENT-FAULT occurs or you kill Legion's processes, please remove semaphores in /dev/shm, for example:
![14b24058fbcfe5bf0648f0d7082686a](https://github.com/JIESUN233/Legion/assets/109936863/c80f6453-6eda-4978-8655-3475cf045457)

## Legion Code Structure
To help users understand Legion's implementation, I list the source code structure in this part.
```
Legion\
├─legion_server.py 
├─sampling_server\                                  ## codes of sampling server
└training_backend\                                  ## codes of training backend

Legion\sampling_server\src\
├─cache\                                            ## unified cache
├─engine\                                           ## pipelining engine of sampling server 
├─storage                                           ## graph/feature storage, system storage initialization
├─include                                           ## system configurations and hashmap (https://github.com/greg7mdp/parallel-hashmap)
├─main.cu                                           ## main function, will be replaced by a python extention module in the future
└Others

Legion\training_backend\
├─legion_graphsage.py                               ## training backend for graphsage model
├─legion_gcn.py                                     ## training backend for gcn model
├─setup.py                                          ## compiling the training backend
├─ipc_service.cpp ipc_service.h ipc_cuda_kernel.cu  ## inter process communication module for training backend with sampling server
└Others
```


