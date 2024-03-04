# cd dataset_prepare/
# # 创建webgraph依赖环境
# mkdir lib
# mv webgraph-3.5.2.jar lib/
# tar -xzvf webgraph-3.6.8-deps.tar.gz -C lib

# # 以webgraph网站中的uk-union数据集为例
# mkdir ukunion

# wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.graph
# wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.properties

# java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph uk-union-2006-06-2007-05-underlying ukunion/ukunion-edgelist.txt
java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph uk-union ukunion/ukunion-edgelist.txt

mkdir xtrapulp_result
# 生成legion格式的bin文件 以及 xtrapulp input格式
g++ gen_legion_xtrapulp_fomat.cpp -o gen_legion_xtrapulp_fomat
./gen_legion_xtrapulp_fomat ukunion ukunion-edgelist.txt


# 2.配置xtrapulp环境
wget https://download.open-mpi.org/release/open-mpi/v3.1/openmpi-3.1.0.tar.gz
tar zxf openmpi-3.1.0.tar.gz

cd openmpi-3.1.0
sudo ./configure --prefix=/usr/local/openmp
sudo make
sudo make install

MPI_HOME=/usr/local/openmpi
export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export MANPATH=${MPI_HOME}/share/man:$MANPATH

# sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
# sudo apt-get install mpich libmpich-dev


# # xtrapulp 分图，参考 https://github.com/luoxiaojian/xtrapulp
# # 以下是一个4进程 8分图的例子:
git clone https://github.com/luoxiaojian/xtrapulp.git
mv ukunion_xtraformat xtrapulp/
cd xtrapulp
make
make libxtrapulp

mpirun -n 4 ./xtrapulp ukunion_xtraformat 8 -v 1.03 -l

# # ukunion_xtraformat.part.8的分图文件的转换为bin格式。第一个arg是生成后的文件名，第二个是该数据集的node数量
cd ../
sudo g++ xtra_part_to_bin.cpp -o xtra_part_to_bin
sudo ./xtra_part_to_bin xtrapulp/ukunion_xtraformat.parts.8 133633040
