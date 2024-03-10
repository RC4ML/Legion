# This file elaborates more on dataset preparing

## Datasets Statistics

| Datasets | PR | PA | CO | UKS | UKL | CL |
| --- | --- | --- | --- | --- | --- | --- |
| #Vertices | 2.4M | 111M | 65M | 133M | 0.79B | 1B |
| #Edges | 120M | 1.6B | 1.8B | 5.5B | 47.2B | 42.5B |
| Feature Size | 100 | 128 | 256 | 256 | 128 | 128 |
| Topology Storage | 640MB | 6.4GB | 7.2GB | 22GB | 189GB | 170GB |
| Feature Storage | 960MB | 56GB | 65GB | 136GB | 400GB | 512GB |
| Class Number | 47 | 2 | 2 | 2 | 2 | 2 |

## Legion Format
Take uk-union as an example
Edge: uk-union/edge_src,  uk-union/edge_dst
feature:  uk-union/features
label: uk-union/labels
partition:  uk-union/partition_8

## Customize your datasets
```
cd dataset/
```
Creature enviroments for webgraph
```
mkdir lib
mv webgraph-3.5.2.jar lib/
tar -xzvf webgraph-3.6.8-deps.tar.gz -C lib
```
Take uk-union for example
```
mkdir ukunion
cd ukunion
wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.graph
wget http://data.law.di.unimi.it/webdata/uk-union-2006-06-2007-05/uk-union-2006-06-2007-05-underlying.properties

# java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph uk-union-2006-06-2007-05-underlying ukunion/ukunion-edgelist.txt
java -cp "lib/*" it.unimi.dsi.webgraph.ArcListASCIIGraph uk-union ukunion/ukunion-edgelist.txt

mkdir xtrapulp_result
# Generate legion-format bin including edge_src, edge_dst, and xtrapulp-format data for graph partitioning
g++ gen_legion_xtrapulp_fomat.cpp -o gen_legion_xtrapulp_fomat
./gen_legion_xtrapulp_fomat ukunion ukunion-edgelist.txt

```

# 2. Graph partitioning
## Install MPI
```
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

# or the instructions in the following
# sudo apt-get install openmpi-bin openmpi-doc libopenmpi-dev
# sudo apt-get install mpich libmpich-dev

```
## Partioning using xtrapulp
refer to https://github.com/luoxiaojian/xtrapulp
An example of using 4 processes to partition graph into 8 parts:
```
git clone https://github.com/luoxiaojian/xtrapulp.git
mv ukunion_xtraformat xtrapulp/
cd xtrapulp
make
make libxtrapulp

mpirun -n 4 ./xtrapulp ukunion_xtraformat 8 -v 1.03 -l

# Convert ukunion_xtraformat.part.8 into legion-format input
sudo g++ xtra_part_to_bin.cpp -o xtra_part_to_bin
sudo ./xtra_part_to_bin xtrapulp/ukunion_xtraformat.parts.8 133633040
```