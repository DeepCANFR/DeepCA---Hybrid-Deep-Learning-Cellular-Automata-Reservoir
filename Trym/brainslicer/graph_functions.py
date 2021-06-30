
from copy import deepcopy

def copy_graph_genome(graph_genome, new_name):
    graph_copy = deepcopy(graph_genome)
    graph_copy["identifier"] = new_name
    return graph_copy

def find_nr_of_nodes(graph_genome):
    nr_of_nodes = 0
    if "graphs" in graph_genome:
        for graph in graph_genome["graphs"]:
            sub_graph_genome = graph_genome["graphs"][graph]
            nr_of_nodes += find_nr_of_nodes(sub_graph_genome)
        return nr_of_nodes 
    elif "nodes" in graph_genome:
        nodes = graph_genome["nodes"]
        nr_of_nodes = len(nodes)
        return nr_of_nodes



def set_node_value_in_genome(graph_genome, map_list, value):
    if "graphs" in graph_genome:
        sub_graphs = graph_genome["graphs"]
        target_graph = sub_graphs[map_list[0]]
        set_node_value_in_genome(target_graph, map_list[1:], value)
    elif "nodes" in graph_genome:
        nodes = graph_genome["nodes"]
        target_node = nodes[map_list[0]]
        target_node[map_list[1]] = value
    else:
        print("Could not find target, please check if location of node is correct")
        print("Current map is: ", map_list)
        sys.exit(0)


def create_folder_tree_names_from_genome(graph_genome, folder_name = ""):
    if "identifier" in graph_genome:
        if folder_name == "":
            full_folder_name = graph_genome["identifier"]
        else:
            full_folder_name = os.path.join(folder_name, graph_genome["identifier"])
    else:
        full_folder_name = folder_name

    if hasattr(graph_genome,'items'):
        for k, v in graph_genome.items():
            if k == "parameters": 
                yield full_folder_name
            if isinstance(v, dict):
                for result in create_folder_tree_names_from_genome(v, full_folder_name):
                    yield result

def find_node_parameters(graph_genome, identifier = ""):
    # based of: https://stackoverflow.com/questions/9807634/find-all-occurrences-of-a-key-in-nested-dictionaries-and-lists
    
    if "identifier" in graph_genome:
        if identifier == "":
            full_node_identifier = graph_genome["identifier"]
        else:
            full_node_identifier = identifier +"-"+ graph_genome["identifier"]
    else:
        full_node_identifier = identifier

    if hasattr(graph_genome,'items'):
        for k, v in graph_genome.items():
            if k == "parameters": 
                yield {"identifier":full_node_identifier, "parameters":v}
            if isinstance(v, dict):
                for result in find_node_parameters(v, full_node_identifier):
                    yield result

def find_node_connections(graph_genome, identifier = ""):
    if "identifier" in graph_genome:
        if identifier == "":
            full_node_identifier = graph_genome["identifier"]
        else:
            full_node_identifier = identifier +"-"+ graph_genome["identifier"]
    else:
        full_node_identifier = identifier
        
    if hasattr(graph_genome, 'items'):
        for k, v in graph_genome.items():
            if k == "connections":
                for connection in v:
                    
                    out_node = full_node_identifier + "-" + connection[0][0]
                    in_node = full_node_identifier + "-" + connection[0][1]

                    out_connection = deepcopy(connection)
                    out_connection[0][0] = out_node
                    out_connection[0][1] = in_node
                    yield out_connection
            if isinstance(v, dict):
                for result in find_node_connections(v, full_node_identifier):
                    yield result

                    
