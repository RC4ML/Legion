default: main

cuda:
	g++ -shared -o libhelper_multiprocess.so helper_multiprocess.cpp -I . -fPIC	
	nvcc -o libgpu_kernel.so \
	-std=c++17 -arch=sm_80 \
	--extended-lambda -shared \
	-Xcompiler -fPIC -lpthread \
	--expt-relaxed-constexpr \
	-I ./include -I ./pcm/src \
	GPUCache.cu GPUMemoryPool.cu \
	Kernels.cu CUDA_IPC_Service.cu \
	GPU_Memory_Node_Storage.cu \
	GPU_Memory_Graph_Storage.cu \
	GPUGraphStore.cu Operator.cu \
	Server.cu\
	 -L . -lhelper_multiprocess  -L ./build/lib -lpcm

main:
	g++ -I . main.cpp -fPIC -O2 \
	-L . -L /usr/local/cuda/lib64 \
	-lcudart  \
	-o legion \
	-lhelper_multiprocess -lgpu_kernel\
	 -I ./src -L ./build/lib -lpcm
