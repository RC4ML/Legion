cd sampling_server && \
make -j 8 && \
python setup.py build_ext --inplace && \
cd .. 
cd training_backend && \
python setup.py install \ 
cd ..
