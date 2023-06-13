#ifndef GRPAHLEARN_CORE_GRAPH_GPUSTORAGE_TYPES_H
#define GRPAHLEARN_CORE_GRAPH_GPUSTORAGE_TYPES_H
#include <cstdint>
#include <vector>
struct BuildInfo{
    //device                                                                                                                        ``
    std::vector<int32_t> shard_to_partition;
    std::vector<int32_t> shard_to_device;
    int32_t partition_count;
    //training set
    std::vector<int32_t> training_set_num;
    std::vector<std::vector<int32_t>> training_set_ids;
    std::vector<std::vector<int32_t>> training_labels;
    //validation set
    std::vector<int32_t> validation_set_num;
    std::vector<std::vector<int32_t>> validation_set_ids;
    std::vector<std::vector<int32_t>> validation_labels;
    //testing set
    std::vector<int32_t> testing_set_num;
    std::vector<std::vector<int32_t>> testing_set_ids;
    std::vector<std::vector<int32_t>> testing_labels;
    //features
    int32_t total_num_nodes;
    int32_t int_attr_len;
    int32_t float_attr_len;
    int64_t* host_int_attrs;//allocated by cudaHostAlloc
    float* host_float_attrs;//allocated by cudaHostAlloc

    //bam params
    uint32_t        cudaDevice;
    uint64_t        cudaDeviceId;
    const char*     blockDevicePath;
    const char*     controllerPath;
    uint64_t        controllerId;
    uint32_t        adapter;
    uint32_t        segmentId;
    uint32_t        nvmNamespace;
    bool            doubleBuffered;
    size_t          numReqs;
    size_t          numPages;
    size_t          startBlock;
    bool            stats;
    const char*     output;
    size_t          numThreads;
    uint32_t        domain;
    uint32_t        bus;
    uint32_t        devfn;
    uint32_t n_ctrls;
    size_t blkSize;
    size_t queueDepth;
    size_t numQueues;
    size_t pageSize;
    uint64_t numElems;
    bool random;
    uint64_t ssdtype;

    //csr
    // std::vector<std::vector<int64_t>> csr_node_index;
    // std::vector<std::vector<int32_t>> csr_dst_node_ids;
    int64_t* csr_node_index;
    int32_t* csr_dst_node_ids;
    // std::vector<char> partition_index;
    // std::vector<int32_t> partition_offset;
    int64_t cache_edge_num;
    int64_t total_edge_num;
    //train
    int32_t epoch;
    int32_t raw_batch_size;
};

#endif