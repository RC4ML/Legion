import subprocess
import re
import networkx as nx
import os
import sys
import argparse 
import pandas as pd
import numpy as np

def parse_topo_output(output):
    """
    Parses the output from `nvidia-smi topo -m` to extract the NVLink connections between GPUs.
    This function is adjusted based on your provided example output.
    """
    connections = []
    lines = output.splitlines()
    gpu_lines = [line for line in lines if line.startswith("GPU")]
    for i, line in enumerate(gpu_lines):
        elements = line.split()
        for j, elem in enumerate(elements[1:], start=0):  # Start from the first GPU column
            if elem.startswith("NV"):
                connections.append((i, j))
    return connections

def get_nvlink_topology():
    # Execute the `nvidia-smi topo -m` command to get the topology matrix
    result = subprocess.run(['nvidia-smi', 'topo', '-m'], stdout=subprocess.PIPE, text=True)
    connections = parse_topo_output(result.stdout)
    return connections

def find_largest_fully_connected_group(G):
    """
    Finds the largest fully connected group (clique) in the graph and returns its size
    and the list of such groups if there are multiple of the same size.
    """
    cliques = list(nx.find_cliques(G))
    max_size = max(len(clique) for clique in cliques) if cliques else 1
    max_cliques = [clique for clique in cliques if len(clique) == max_size]
    return max_size, max_cliques

if __name__ == "__main__":
    
    cur_path = sys.path[0]
    argparser = argparse.ArgumentParser("Graph Partitioning.")
    argparser.add_argument('--dataset_path', type=str, default='dataset')
    argparser.add_argument('--dataset_name', type=str, default='ukunion')
    argparser.add_argument('--processes_number', type=int, default=4)
    argparser.add_argument('--gpu_num', type=int, default=2)
    args = argparser.parse_args()
    
    
    if args.dataset_name == "products":
        path =  args.dataset_path + "/products/"
        vertices_num = 2449029
        edges_num = 123718280
        features_dim = 100
        train_set_num = 196615
        valid_set_num = 39323
        test_set_num = 2213091
    elif args.dataset_name == "paper100m":
        path = args.dataset_path + "/paper100m/"
        vertices_num = 111059956
        edges_num = 1615685872
        features_dim = 128
        train_set_num = 11105995
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "com-friendster":
        path = args.dataset_path + "/com-friendster/"
        vertices_num = 65608366
        edges_num = 1806067135
        features_dim = 256
        train_set_num = 6560836
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "ukunion":
        path = args.dataset_path + "/ukunion/"
        vertices_num = 133633040
        edges_num = 5507679822
        features_dim = 256
        train_set_num = 13363304
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "uk2014":
        path = args.dataset_path + "/uk2014/"
        vertices_num = 787801471
        edges_num = 47284178505
        features_dim = 128
        train_set_num = 78780147
        valid_set_num = 100000
        test_set_num = 100000
    elif args.dataset_name == "clueweb":
        path = args.dataset_path + "/clueweb/"
        vertices_num = 955207488
        edges_num = 42574107469
        features_dim = 128
        train_set_num = 95520748
        valid_set_num = 100000
        test_set_num = 100000
    else:
        print("invalid dataset path")
        exit    
    
    connections = get_nvlink_topology()
    G = nx.Graph()
    G.add_edges_from(connections)
    group_size, fully_connected_groups = find_largest_fully_connected_group(G)
    if int(args.gpu_num/group_size) > 1:
        partition_command = [
            "mpirun",
            "-n", "4",
            "./dataset/xtrapulp/xtrapulp",
            "./dataset/xtrapulp/" + args.dataset_name+"_xtraformat",
            str(int(args.gpu_num/group_size)),
            "-v", "1.03",
            "-l"
        ]
        print(partition_command)
        result = subprocess.run(partition_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        file_path = "./dataset/xtrapulp/"+args.dataset_name+"_xtraformat.parts."+str(int(args.gpu_num/group_size))
        df = pd.read_csv(file_path, header=None, delimiter="\s+")
        data = df.to_numpy()
        data = data.astype(np.int32)
        data.tofile('partition')
        # print(data)

        move_command = [
            "mv",
            "partition",
            "./dataset/"+args.dataset_name+"/"
        ]
        print(move_command)
        result2 = subprocess.run(move_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)     
        print("STDERR:", result2.stderr)
    