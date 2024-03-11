cd sampling_server && \
make clean && make -j 8 && \
cd .. && \
cd training_backend && \
python setup.py install && \ 
cd ..
