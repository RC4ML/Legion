cd sampling_server && \
make clean && make -j 8 && \
cd .. && \
cd training_backend && \
export CUDA_HOME=/usr/local/cuda && \
python setup.py install && \ 
cd ..
