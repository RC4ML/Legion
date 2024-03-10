
extern "C"
void* d_alloc_space(int64_t num_bytes) {
    void *ret;
	cudaMalloc(&ret, num_bytes);
	cudaCheckError();
    return ret;
}

extern "C"
void* d_alloc_space_managed(unsigned int num_bytes) {
    void *ret;
	cudaMallocManaged(&ret, num_bytes);
	cudaCheckError();
    return ret;
}

extern "C"
void d_copy_2_h(void* h_ptr, void* d_ptr, unsigned int num_bytes){
	cudaMemcpy(h_ptr, d_ptr, num_bytes, cudaMemcpyDeviceToHost);
	cudaCheckError();
}


extern "C"
void SetGPUDevice(int32_t shard_id){
	cudaSetDevice(shard_id);
	cudaCheckError();
}

extern "C"
int32_t GetGPUDevice(){
	int32_t dev_id;
	cudaGetDevice(&dev_id);
	return dev_id;
}

extern "C"
void d_free_space(void* d_ptr){
	cudaFree(d_ptr);
}


extern "C"
void* host_alloc_space(unsigned int num_bytes) {
    void* host_ptr;
	void* ret;
	cudaHostAlloc(&host_ptr, num_bytes, cudaHostAllocMapped);
	cudaHostGetDevicePointer(&ret, host_ptr, 0);
	cudaCheckError();
    return ret;
}