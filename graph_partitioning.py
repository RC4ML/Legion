import subprocess
import re
import networkx as nx

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
    connections = get_nvlink_topology()
    G = nx.Graph()
    G.add_edges_from(connections)
    group_size, fully_connected_groups = find_largest_fully_connected_group(G)
    
    if fully_connected_groups or group_size == 1:
        print(f"Largest group size with fully connected GPUs: {group_size}")
        