#include <iostream>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <fcntl.h>
#include <fstream>
#include <stdio.h>
#include <unistd.h>
#include <errno.h>
#include <vector>
#include <map>
using namespace std;

int64_t node_nums = 133633040;
int64_t edge_nums = 5507679822;

#define ten_e 1000000000
#define one_e 100000000

vector<vector<int64_t>> vc;
vector<int64_t> indptr;
vector<int> indices;
vector<int> edges;

int64_t lines = 0;
int cnt = 1;
int reid = 0;
int64_t qq = one_e;
int64_t gl_nums = 0;
bool flag[300000000];
int32_t* idmap;

void read(std::string &file_path) {
	int fd = open(file_path.c_str(), O_RDONLY);
	int64_t buf_size = lseek(fd, 0, SEEK_END);
	char* buf = (char*)mmap(NULL, buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	const char* buf_end = buf + buf_size;
	int src, dst;
	std::string str = "";
	while(buf < buf_end) {
		if(*buf == '\t'){
			src = stoi(str);
			str = "";
			buf++;
			continue;
		}
		if (*buf == '\n') {
            lines++;
            if(lines%qq==0){
                cout<<"loaded "<<cnt*qq<<" edges"<<endl;
                cnt++;
            }
			dst = stoi(str);
            
            flag[src] = true;
            flag[dst] = true;
            
			str = "";
			++buf;
			continue;
		}
		str += *buf++;
	}
	close(fd);
	return ;
}

void reoder_read(std::string &file_path) {
	int fd = open(file_path.c_str(), O_RDONLY);
	int64_t buf_size = lseek(fd, 0, SEEK_END);
	char* buf = (char*)mmap(NULL, buf_size, PROT_READ, MAP_PRIVATE, fd, 0);
	const char* buf_end = buf + buf_size;
	int src, dst;
	std::string str = "";
	while(buf < buf_end) {
		if(*buf == '\t'){
			src = stoi(str);
			str = "";
			buf++;
			continue;
		}
		if (*buf == '\n') {
            lines++;
            if(lines%qq==0){
                cout<<"loaded "<<cnt*qq<<" edges"<<endl;
                cnt++;
            }
			dst = stoi(str);

            if(src != dst){
                int new_src = idmap[src];
                int new_dst = idmap[dst];
                vc[new_src].push_back(new_dst);
                edges.push_back(new_src);
                edges.push_back(new_dst);
                // vc[idmap[src]].push_back(idmap[dst]);
            }
            
			str = "";
			++buf;
			continue;
		}
		str += *buf++;
	}
	close(fd);
	return ;
}

int main(int argc, char *argv[]) 
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <file_dir> <file_name>" << std::endl;
        return 1;
    }
    std::string file_dir = argv[1];
    std::string file_name = argv[2];
    std::string file_path = file_dir + "/" + file_name;
    // string file_path = "ukunion/ukunion-edgelist.txt";

    memset(flag,false,sizeof(flag));
    idmap = (int32_t*)malloc(int64_t(int64_t(300000000) * sizeof(int32_t)));

    cout<<"edges loading:"<<endl;
    read(file_path);
    
    cout<<"flag find:"<<endl;
    for(int i=0;i<300000000;i++){
        if(flag[i] == true){
            idmap[i] = reid++;
            gl_nums++;
        }
    }
    
    cout<<"nodes: "<<gl_nums<<endl;
    cout<<"now reid: "<<reid<<endl;

    cout<<"starting reorder:"<<endl;
    vc.resize(gl_nums);
    lines = 0;
    cnt = 1;
    reoder_read(file_path);

    //loading indptr and indices
    cout<<"loading indptr and indices: "<<endl;
    indptr.push_back(0);
    for(int i=0;i<vc.size();i++){
        int64_t now_size = vc[i].size();
        indptr.push_back(now_size+indptr.back());
        for(int64_t j=0;j<now_size;j++){
            indices.push_back(vc[i][j]);
        }
    }

    cout<<"indptr last value: "<<indptr[indptr.size()-1]<<endl;
    cout<<"indices last value: "<<indices[indices.size()-1]<<endl;
    // cout<<"lines: "<<lines<<endl;

    cout<<"starting writing:"<<endl;
    std::string data_edge_src_name = file_dir + "/" + file_dir + "_edge_src" ;
    std::string data_edge_dst_name = file_dir + "/" + file_dir + "_edge_dst" ;
    std::string data_xtraformat_name = file_dir + "_xtraformat" ;

    ofstream out_indptr(data_edge_src_name);
    ofstream out_indices(data_edge_dst_name);
    ofstream out(data_xtraformat_name);
    
    // ofstream gnnlab_indptr("indptr.bin");

    for(int64_t i=0;i<indptr.size();i++){
        out_indptr.write((char*)&indptr[i], sizeof(int64_t));
    }
    // for(int64_t i=0;i<indptr.size();i++){
    //     gnnlab_indptr.write((char*)&indptr[i], sizeof(int));
    // }
    for(int64_t i=0;i<indices.size();i++){
        out_indices.write((char*)&indices[i], sizeof(int));
    }

    for(int64_t i=0;i<edges.size();i++){
        out.write((char*)&edges[i], sizeof(int));
    }

    cout<<"convert finished!"<<endl;
    out_indptr.close();
    // gnnlab_indptr.close();
    out_indices.close();
    // out.close();
    return 0;
}
