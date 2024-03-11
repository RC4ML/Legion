cd dataset/
# # create webgraph environment
mkdir lib
cp webgraph-3.5.2.jar lib/
tar -xzvf webgraph-3.6.8-deps.tar.gz -C lib

mkdir ukunion
cd ukunion
wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.graph
wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.properties
cd ..
java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph ukunion/uk-union-2006-06-2007-05-underlying ukunion/ukunion-edgelist.txt

mkdir xtrapulp_result
# generate legion-format edge_src edge_dst, and the input of xtrapulp
g++ gen_legion_xtrapulp_fomat.cpp -o gen_legion_xtrapulp_fomat
./gen_legion_xtrapulp_fomat ukunion ukunion-edgelist.txt
# generate training sets, validation sets, and test sets
python gen_sets.py --dataset_name ukunion

# 2.If you don't have mpi, install mpi first.

# wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.0.tar.gz
# tar zxf openmpi-3.1.0.tar.gz

# cd openmpi-3.1.0
# sudo ./configure --prefix=/usr/local/openmp
# sudo make
# sudo make install

# MPI_HOME=/usr/local/openmpi
# export PATH=${MPI_HOME}/bin:$PATH
# export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
# export MANPATH=${MPI_HOME}/share/man:$MANPATH

# sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
# sudo apt-get install mpich libmpich-dev


# # install xtrapulp, refer to https://github.com/luoxiaojian/xtrapulp
git clone https://github.com/luoxiaojian/xtrapulp.git
mv ukunion_xtraformat xtrapulp/
cd xtrapulp
make
make libxtrapulp
cd ../../
