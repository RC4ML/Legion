#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "server.h"
#include <stdlib.h>
#include <iostream>

int Run(const std::vector<int>& fanout, int gpu_number, int in_memory_mode, int cache_mode){
    std::cout<<"Start Sampling Server\n";
    Server* server = NewGPUServer();
    server->Initialize(gpu_number, fanout, in_memory_mode);//gpu number, default 1; in memory, default true
    server->PreSc(cache_mode);//cache aggregate mode, default 0
    server->Run();
    server->Finalize();
    return 0;
}

namespace py = pybind11;

PYBIND11_MODULE(sampling_server, m) {
    m.doc() = "pybind11 plugin"; 
    m.def("Run", &Run, "Run Sampling Server");
}
