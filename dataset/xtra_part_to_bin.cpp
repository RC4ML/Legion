#include <iostream>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <vector>
#include <chrono>
#include <iomanip>
#include <ofstream>
#include <immintrin.h>

std::vector<int> nodes;

int64_t node_index = 0;

void mmap_read(std::string &indices_file){
    int fd = open(indices_file.c_str(), O_RDONLY);
    int64_t buf_len = lseek(fd, 0, SEEK_END);
    const int *buf = (int *)mmap(NULL, buf_len, PROT_READ, MAP_PRIVATE, fd, 0);
    const int* buf_end = buf + buf_len/sizeof(int);
    int temp;
    while(buf < buf_end){
        temp = *buf;
        nodes[node_index++] = temp;
        buf++;
    }
    close(fd);
    return;
}

int main(int argc, char *argv[]) 
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <part_file> <node_nums>" << std::endl;
        return 1;
    }
    std::string part_file = argv[1];

    int64_t node_nums = std::atoll(argv[2]);
    nodes.resize(node_nums);

    mmap_read(part_file);
    std::string part_bin_file = part_file + ".bin";
    ofstream out(part_bin_file);

    for(int64_t i=0;i<nodes.size();i++){
        out.write((char*)&nodes[i], sizeof(int));
    }

    return 0;
}