#ifndef SERVER_H
#define SERVER_H
#include <vector>

struct RunnerParams {
    int device_id;
    std::vector<int> fanout;
    void* cache;
    void* graph;
    void* feature;
    void* env;
    int global_batch_id;
    bool in_memory;
};

class Server {
public:
    virtual void Initialize(int global_shard_count, std::vector<int> fanout, int in_memory_mode) = 0;
    virtual void PreSc(int cache_agg_mode) = 0;
    virtual void Run() = 0;
    virtual void Finalize() = 0;
};
Server* NewGPUServer();

class Runner {
public:
    virtual void Initialize(RunnerParams* params) = 0;
    virtual void InitializeFeaturesBuffer(RunnerParams* params) = 0;
    virtual void RunPreSc(RunnerParams* params) = 0;
    virtual void RunOnce(RunnerParams* params) = 0;
    virtual void Finalize(RunnerParams* params) = 0;
};
Runner* NewGPURunner();

#endif