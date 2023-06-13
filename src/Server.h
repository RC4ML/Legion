#ifndef SERVER_H
#define SERVER_H
#include <vector>

#include <iostream>
#ifdef _MSC_VER
#include <windows.h>
#include "windows/windriver.h"
#else
#include <unistd.h>
#include <signal.h>
#endif
#include <math.h>
#include <iomanip>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
// #include <pthread.h>
#include <assert.h>
#include "./pcm/src/pcm-pcie.h"
// #include "cpucounters.h"

#define PCM_DELAY_DEFAULT 1.0 // in seconds
#define PCM_DELAY_MIN 0.015 // 15 milliseconds is practical on most modern CPUs

using namespace std;

bool events_printed = false;

// #include "zerocp.h"

IPlatform *IPlatform::getPlatform(PCM *m, bool csv, bool print_bandwidth, bool print_additional_info, uint32 delay)
{
    switch (m->getCPUModel()) {
        case PCM::ICX:
        case PCM::SNOWRIDGE:
            return new WhitleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
        case PCM::SKX:
            return new PurleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
        case PCM::BDX_DE:
        case PCM::BDX:
        case PCM::KNL:
        case PCM::HASWELLX:
            return new GrantleyPlatform(m, csv, print_bandwidth, print_additional_info, delay);
        case PCM::IVYTOWN:
        case PCM::JAKETOWN:
            return new BromolowPlatform(m, csv, print_bandwidth, print_additional_info, delay);
        default:
          return NULL;
    }
}

class PCM_Monitor {
public:
    void Init();
    void Start();
    void Stop();
    void Print();
    std::vector<uint64_t> GetCounter();
    
private:
    std::vector<uint64_t> counter_;
    IPlatform* platform_;
    vector<eventGroup_t> *eventGroups_;
    PCM * m_;
};

// #endif

void PCM_Monitor::Init(){

    set_signal_handlers();

    cerr << "\n";
    cerr << " Intel(r) Performance Counter Monitor: PCIe Bandwidth Monitoring Utility \n";
    cerr << " This utility measures PCIe bandwidth in real-time\n";
    cerr << "\n";

    // double delay = 1.0;
    bool csv = false;
    bool print_bandwidth = true;
	bool print_additional_info = false;
    // char * sysCmd = NULL;
    // char ** sysArgv = NULL;
    MainLoop mainLoop;

    m_= PCM::getInstance();
    
    platform_ = IPlatform::getPlatform(m_, csv, print_bandwidth,
                                    print_additional_info, 1); // FIXME: do we support only integer delay? ; lgtm [cpp/fixme-comment]
    
    if (!platform_)
    {
        print_cpu_details();
        cerr << "Jaketown, Ivytown, Haswell, Broadwell-DE Server CPU is required for this tool! Program aborted\n";
        exit(EXIT_FAILURE);
    }

    eventGroups_= platform_->getEventGroups();

    platform_->cleanup(); 
    MySleepMs(uint(1000));
}

void PCM_Monitor::Start(){
    printf("Start Count PCIe\n");
    for (auto &evGroup : *eventGroups_){
        m_->programPCIeEventGroup(evGroup);
        platform_->getEventGroup(evGroup, 0);
    }
    for (auto &evGroup : *eventGroups_){
        m_->programPCIeEventGroup(evGroup);
    }
}

void PCM_Monitor::Stop(){
    for (auto &evGroup : *eventGroups_){
        platform_->getEventGroup(evGroup, 1);
    }
    printf("Stop Count PCIe\n");
}

void PCM_Monitor::Print(){
    platform_->printHeader();

    platform_->printEvents();

    platform_->printAggregatedEvents();
}

std::vector<uint64_t> PCM_Monitor::GetCounter(){
    
    return platform_->GetCounter();
}

struct RunnerParams {
    int device_id;
    std::vector<int> fanout;
    void* cache;
    void* graph;
    void* noder;
    void* env;
    int global_batch_id;
    bool in_memory;
};

class Server {
public:
    virtual void Initialize(int global_shard_count) = 0;
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