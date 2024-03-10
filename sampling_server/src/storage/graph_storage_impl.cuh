#pragma once
#ifndef GRAPH_STORAGE_IMPL_H_
#define GRAPH_STORAGE_IMPL_H_

#include <cstdint>
#include <string>
#include <vector>
#include <iostream>
#include <thrust/random/uniform_int_distribution.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>


// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
  

__global__ void assign_memory(int32_t** int32_pptr, int32_t* int32_ptr, int64_t** int64_pptr, int64_t* int64_ptr, int32_t device_id){
    int32_pptr[device_id] = int32_ptr;
    int64_pptr[device_id] = int64_ptr;
}


__global__ void GetNeighborCount(int32_t* QT, int32_t Kg, int32_t Ki, int32_t capacity, int64_t* csr_node_index_cpu, int64_t* neighbor_count){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        int32_t cache_id = QT[thread_idx * Kg + Ki];
        int64_t count = csr_node_index_cpu[cache_id + 1] - csr_node_index_cpu[cache_id];
        neighbor_count[thread_idx] = count;
    }
}

__global__ void TopoFillUp(int32_t* QT, int32_t Kg, int32_t Ki, int32_t capacity, 
                            int64_t* csr_node_index_cpu, int32_t* csr_dst_node_ids_cpu, 
                             int64_t* d_csr_node_index, int32_t* d_csr_dst_node_ids){
    for(int32_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x; thread_idx < capacity; thread_idx += gridDim.x * blockDim.x){
        int32_t cache_id = QT[thread_idx * Kg + Ki];
        int64_t count = csr_node_index_cpu[cache_id + 1] - csr_node_index_cpu[cache_id];
        for(int i = 0; i < count; i++){
            int32_t neighbor_id = csr_dst_node_ids_cpu[csr_node_index_cpu[cache_id] + i];
            int64_t start_off = d_csr_node_index[thread_idx];
            d_csr_dst_node_ids[start_off + i] = neighbor_id;
        }
    }
}



#endif