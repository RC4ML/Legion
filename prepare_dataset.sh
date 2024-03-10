cd dataset/
# # 创建webgraph依赖环境
mkdir lib
mv webgraph-3.5.2.jar lib/
tar -xzvf webgraph-3.6.8-deps.tar.gz -C lib

# # 以webgraph网站中的uk-union数据集为例
mkdir ukunion

wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.graph
wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.properties

java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph uk-union-2006-06-2007-05-underlying ukunion/ukunion-edgelist.txt

mkdir xtrapulp_result
# 生成legion格式的bin文件 以及 xtrapulp input格式
g++ gen_legion_xtrapulp_fomat.cpp -o gen_legion_xtrapulp_fomat
./gen_legion_xtrapulp_fomat ukunion ukunion-edgelist.txt


# 2.Graph partitioning if you don't have mpi, install mpi first.

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


# # xtrapulp partitioning, refer to https://github.com/luoxiaojian/xtrapulp
# # An example of using 4 processes to partition graph into 8 parts:
git clone https://github.com/luoxiaojian/xtrapulp.git
mv ukunion_xtraformat xtrapulp/
cd xtrapulp
make
make libxtrapulp

mpirun -n 4 ./xtrapulp ukunion_xtraformat 8 -v 1.03 -l

# # Convert ukunion_xtraformat.part.8 into legion-format input, arg1: xtraformat output name, arg2: node number of the dataset
cd ../
sudo g++ xtra_part_to_bin.cpp -o xtra_part_to_bin
sudo ./xtra_part_to_bin xtrapulp/ukunion_xtraformat.parts.8 133633040
