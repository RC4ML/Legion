#include "storage_management.cuh"
#include "graph_storage.cuh"
#include "feature_storage.cuh"
#include "ipc_service.h"
#include "cache.cuh"
#include "operator.h"
#include "memorypool.cuh"
#include "server.h"
#include "server_imp.cuh"
// #include "monitor.cuh"
#include "system_config.cuh"

#include <thread>
#include <functional>
#include <chrono>


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

void PreSCLoop(int train_step, Runner* runner, RunnerParams* params){
    for(int i = 0; i < train_step; i++){
        params->global_batch_id = i;
        runner->RunPreSc(params);
    }
    runner->InitializeFeaturesBuffer(params);
} 

void RunnerLoop(int max_step, Runner* runner, RunnerParams* params){
    for(int i = 0; i < max_step; i++){
        params->global_batch_id = i;
        runner->RunOnce(params);
    }
}

class GPUServer : public Server {
public:
    void Initialize(int global_shard_count, std::vector<int> fanout, int in_memory_mode) {
        shard_count_ = global_shard_count;
        // std::cout<<"CUDA Device Count: "<<shard_count_<<"\n";
        if (in_memory_mode){
            std::cout<<"In Memory Mode\n";
        }else{
            std::cout<<"In Disk Mode\n";
        }
        // monitor_ = new PCM_Monitor();
        // monitor_->Init();
        
        StorageManagement* storage_management = new StorageManagement();
        storage_management->Initialze(shard_count_, in_memory_mode);
        graph_              = storage_management->GetGraph();
        feature_            = storage_management->GetFeature();
        cache_              = storage_management->GetCache();
        ipc_env_            = storage_management->GetIPCEnv();

        train_step_         = ipc_env_->GetTrainStep();
        max_step_           = ipc_env_->GetMaxStep();

        runners_.resize(shard_count_);
        params_.resize(shard_count_);

        for(int i = 0; i < shard_count_; i++){
            cudaSetDevice(i);
            RunnerParams* new_params = new RunnerParams();
            new_params->device_id = i;
            for(int j = 0; j < fanout.size(); j++){
                (new_params->fanout).push_back(fanout[j]);
            }
            new_params->cache           = (void*)cache_;
            new_params->graph           = (void*)graph_;
            new_params->feature         = (void*)feature_;
            new_params->env             = (void*)ipc_env_;
            new_params->global_batch_id = 0;
            new_params->in_memory       = 1;
            params_[i]                  = new_params;
            Runner* new_runner          = NewGPURunner();
            runners_[i]                 = new_runner;
            runners_[i]->Initialize(params_[i]);
        }
    }

    void PreSc(int cache_agg_mode) {
        // std::cout<<"Start Pre-sampling"<<std::endl;
        // monitor_->Start();
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        for(int i = 0; i < shard_count_; i++){
            Runner* runner = runners_[i];
            RunnerParams* params = params_[i];
            std::thread th(&PreSCLoop, train_step_, runner, params);
            presc_thread_pool_.push_back(std::move(th));
        }
        for(auto &th : presc_thread_pool_){
            th.join();
        }

        // monitor_->Stop();
        std::vector<uint64_t> counters(2,0); //=  monitor_->GetCounter();
        double t = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();

        cache_->CandidateSelection(cache_agg_mode, feature_, graph_);
        cache_->CostModel(cache_agg_mode, feature_, graph_, counters, train_step_);
        cache_->FillUp(cache_agg_mode, feature_, graph_);
        // cache_->HybridInit(feature_, graph_);

        std::cout<<"Preprocessing cost: "<<t<<" s\n";

        std::cout<<"System is ready for serving\n";
    }

    void Run() {
        // monitor_->Start();
        // std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        for(int i = 0; i < shard_count_; i++){
            Runner* runner = runners_[i];
            RunnerParams* params = params_[i];
            std::thread th(&RunnerLoop, max_step_, runner, params);
            train_thread_pool_.push_back(std::move(th));
        }
        for(auto &th : train_thread_pool_){
            th.join();
        }
        // double t = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - t1).count();
        // monitor_->Stop();
        // std::vector<uint64_t> counters = monitor_->GetCounter();
        // std::cout<<counters[0]<<" "<<counters[1]<<"\n";
        // std::cout<<"sampling cost: "<<t<<" s\n";
    }

    void Finalize() {
        for(int i = 0; i < shard_count_; i++){
            runners_[i]->Finalize(params_[i]);
        }
        graph_->Finalize();
        feature_->Finalize();
        // cache_->Finalize();
        ipc_env_->Finalize();
        std::cout<<"Server Stopped\n";
    }
private:

    GraphStorage* graph_;
    FeatureStorage* feature_;
    UnifiedCache* cache_;
    IPCEnv* ipc_env_;
    // PCM_Monitor* monitor_;

    int shard_count_;
    int train_step_;
    int max_step_;

    std::vector<std::thread> presc_thread_pool_;
    std::vector<std::thread> train_thread_pool_;
    std::vector<Runner*> runners_;
    std::vector<RunnerParams*> params_;
};

Server* NewGPUServer(){
    return new GPUServer();
}

class GPURunner : public Runner {
public:
    void Initialize(RunnerParams* params) override {
        cudaSetDevice(params->device_id);
        local_dev_id_           = params->device_id;
        UnifiedCache* cache     = (UnifiedCache*)(params->cache);
        GraphStorage* graph     = (GraphStorage*)(params->graph);
        FeatureStorage* feature   = (FeatureStorage*)(params->feature);
        IPCEnv* env             = (IPCEnv*)(params->env);

        /*initialize GPU environment*/
        streams_.resize(INTRABATCH_CON);
        for(int i = 0; i < INTRABATCH_CON; i++){
            cudaStreamCreate(&streams_[i]);
        }

        /*dag params analysis*/
        int batch_size          = env->GetRawBatchsize();
        int max_ids_num         = batch_size;
        std::vector<int32_t> max_num_per_hop;
        int hop_num             = (params->fanout).size();
        max_num_per_hop.resize(hop_num);
        max_num_per_hop[0]      = batch_size * (params->fanout)[0];
        for(int i = 1; i < hop_num; i++){
            max_num_per_hop[i]  = max_num_per_hop[i - 1] * (params->fanout)[i];
        }
        for(int i = 0; i < hop_num; i++){
            max_ids_num += max_num_per_hop[i];
        }
        num_ids_ = max_ids_num;

        op_num_ = (hop_num + 1) * INTRABATCH_CON + 1;
        op_factory_.resize(op_num_);
        op_factory_[0] = NewBatchGenerateOP(0);
        op_factory_[1] = NewCacheLookupOP(1);
        op_factory_[2] = NewSSDIOSubmitOP(2);
        for(int i = 0; i < hop_num; i++){
            op_factory_[INTRABATCH_CON * i + 3] = NewRandomSampleOP(INTRABATCH_CON * i + 3);
            op_factory_[INTRABATCH_CON * i + 4] = NewCacheLookupOP(INTRABATCH_CON * i + 4);
            op_factory_[INTRABATCH_CON * i + 5] = NewSSDIOSubmitOP(INTRABATCH_CON * i + 5);
        };
        op_factory_[op_num_ - 1] = NewSSDIOCompleteOP(op_num_ - 1);

        /*buffer allocation*/
        int interbatch_concurrency = INTERBATCH_CON;
        interbatch_concurrency_ = interbatch_concurrency;

        int total_num_nodes = feature->TotalNodeNum();
        cache->InitializeCacheController(local_dev_id_, total_num_nodes);/*control cache memory by current actor*/

        memorypool_                     = new MemoryPool(interbatch_concurrency);
        int32_t* cache_search_buffer    = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetCacheSearchBuffer(cache_search_buffer);
        uint32_t* accessed_map          = (uint32_t*)d_alloc_space(int64_t((int64_t(total_num_nodes / 32)  + 1)* sizeof(uint32_t)));
        memorypool_->SetAccessedMap(accessed_map);
        int32_t* position_map           = (int32_t*)d_alloc_space(int64_t(int64_t(total_num_nodes) * sizeof(int32_t)));
        memorypool_->SetPositionMap(position_map);
        int32_t* agg_src_ids            = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetAggSrcId(agg_src_ids);
        int32_t* agg_dst_ids            = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetAggDstId(agg_dst_ids);
        char* tmp_part_ind              = (char*)d_alloc_space(num_ids_ * sizeof(char));
        memorypool_->SetTmpPartIdx(tmp_part_ind);
        int32_t* tmp_part_off           = (int32_t*)d_alloc_space(num_ids_ * sizeof(int32_t));
        memorypool_->SetTmpPartOff(tmp_part_off);


        int32_t float_feature_len = feature->GetFloatFeatureLen();
        float_feature_len_ = float_feature_len;
        env->InitializeSamplesBuffer(batch_size, num_ids_, float_feature_len_, local_dev_id_, interbatch_concurrency);
        current_pipe_ = 0;
        for(int i = 0; i < INTERBATCH_CON; i++){
          memorypool_->SetSampledIds(env->GetIds(local_dev_id_, i), i);
          memorypool_->SetLabels(env->GetLabels(local_dev_id_, i), i);
          memorypool_->SetAggSrcOf(env->GetAggSrc(local_dev_id_, i), i);
          memorypool_->SetAggDstOf(env->GetAggDst(local_dev_id_, i), i);
          memorypool_->SetNodeCounter(env->GetNodeCounter(local_dev_id_, i), i);
          memorypool_->SetEdgeCounter(env->GetEdgeCounter(local_dev_id_, i), i);
        }
        
        events_.resize(op_num_);
        op_params_.resize(op_num_);

        bool in_memory = params->in_memory;

        for(int i = 0; i < op_num_; i++){
            op_params_[i] = new OpParams();
            op_params_[i]->device_id    = local_dev_id_;
            op_params_[i]->stream       = (streams_[i%INTRABATCH_CON]);
            cudaEventCreate(&events_[i]);
            op_params_[i]->event        = (events_[i]);
            op_params_[i]->memorypool   = memorypool_;
            op_params_[i]->cache        = cache;
            op_params_[i]->graph        = graph;
            op_params_[i]->feature      = feature;
            op_params_[i]->env          = env;
            op_params_[i]->in_memory    = in_memory;
            op_params_[i]->hop_num      = hop_num;
        }

        for(int i = 0; i < hop_num; i++){
            op_params_[INTRABATCH_CON * i + INTRABATCH_CON]->neighbor_count = (params->fanout)[i];
        }
    }

    void InitializeFeaturesBuffer(RunnerParams* params) override {
        UnifiedCache* cache     = (UnifiedCache*)(params->cache);
        int32_t num_ids         = int32_t((cache->MaxIdNum(local_dev_id_)) * 1.2);
        IPCEnv* env             = (IPCEnv*)(params->env);
        env->InitializeFeaturesBuffer(0, num_ids, float_feature_len_, local_dev_id_, interbatch_concurrency_);
        for(int i = 0; i < interbatch_concurrency_; i++){
          memorypool_->SetFloatFeatures(env->GetFloatFeatures(local_dev_id_, i), i);
        }
    }

    void RunPreSc(RunnerParams* params) override {
        cudaSetDevice(local_dev_id_);
        int32_t batch_id = params->global_batch_id;
        memorypool_->SetCurrentMode(0);
        memorypool_->SetIter(batch_id);
        for(int i = 0; i < op_num_; i+=INTRABATCH_CON){
            op_params_[i]->is_presc = true;
            op_factory_[i]->run(op_params_[i]);
        }
        bool is_ready = false;
        while(!is_ready){
            if(!(cudaEventQuery((op_params_[op_num_-1]->event)) == cudaErrorNotReady)){
                is_ready = true;
            }
        }
    }

    void RunOnce(RunnerParams* params) override {
        cudaSetDevice(local_dev_id_);
        IPCEnv* env = (IPCEnv*)(params->env);
        int32_t batch_id = params->global_batch_id;
        mode_ = env->GetCurrentMode(batch_id);
        memorypool_->SetCurrentMode(mode_);
        memorypool_->SetIter(env->GetLocalBatchId(batch_id));
        env->IPCWait(local_dev_id_, current_pipe_);
        
        for(int i = 0; i < op_num_; i++){
            if(i % INTRABATCH_CON >= 1){
                cudaStreamWaitEvent(streams_[i % INTRABATCH_CON], events_[i / INTRABATCH_CON * INTRABATCH_CON], 0);
            }
            op_params_[i]->is_presc = false;
            op_factory_[i]->run(op_params_[i]);
        }
        
        bool is_ready = false;
        while(!is_ready){
            if(!(cudaEventQuery((op_params_[op_num_-1]->event)) == cudaErrorNotReady)){
                is_ready = true;
            }
        }

        env->IPCPost(local_dev_id_, current_pipe_);
        if(batch_id % 1000 == 0 && local_dev_id_ == 0){
            std::cout<<"batch id: "<<batch_id<<"\n";
        }
        current_pipe_ = (current_pipe_ + 1) % interbatch_concurrency_;
        memorypool_ -> SetCurrentPipe(current_pipe_);
    }

    void Finalize(RunnerParams* params) override {
        IPCEnv* env = (IPCEnv*)(params->env);
        env->IPCWait(local_dev_id_, (current_pipe_ + 1) % interbatch_concurrency_);
        cudaSetDevice(local_dev_id_);
        memorypool_->Finalize();
    }

private:

    /*vertex id & feature buffer*/
    int32_t num_ids_;
    int32_t float_feature_len_;

    /*buffers for multi gpu tasks*/
    MemoryPool* memorypool_;

    /*pipeline*/
    int current_pipe_;
    int interbatch_concurrency_;

    /*map to physical device*/
    int local_dev_id_;

    /*mode, training(0), validation(1), testing(2)*/
    int mode_;
    int op_num_;
    
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    std::vector<Operator*> op_factory_;
    std::vector<OpParams*> op_params_;
};

Runner* NewGPURunner(){
    return new GPURunner();
}

