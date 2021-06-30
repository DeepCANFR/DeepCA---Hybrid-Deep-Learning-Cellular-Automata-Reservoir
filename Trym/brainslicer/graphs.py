from .graph_functions import find_node_parameters, find_node_connections, set_node_value_in_genome
import os
import json 
import numpy as np

class DistributedGraph:
    def __init__(self, client, available_classes):
        self.client = client
        self.available_classes = available_classes
        
  
    def construct_distributed_graph(self, graph_genome):
        self.graph_genome = graph_genome
        self.construct_nodes(graph_genome)
        self.connect_nodes(graph_genome)

    def construct_nodes(self, graph_genome = ""):
        if graph_genome == "":
            graph_genome = self.graph_genome

        self.nodes = {}
        for node_data in find_node_parameters(graph_genome):
            identifier = node_data["identifier"]
            parameters = node_data["parameters"]
            parameters["identifier"] = identifier

            node_type = self.available_classes[parameters["type"]]
            self.nodes[identifier] = self.client.submit(node_type, parameters, actor = True)
        
        for identifier in self.nodes:
            self.nodes[identifier] = self.nodes[identifier].result()

    def connect_nodes(self, graph_genome = ""):
        if graph_genome == "":
            graph_genome = self.graph_genome
        futures = []
        self.connections = []
        for connection in find_node_connections(graph_genome):
            out_node_identifier = connection[0][1]
            in_node_identfier = connection[0][0]
            
            out_node = self.nodes[out_node_identifier]
            in_node = self.nodes[in_node_identfier]

            out_node_connection_variable = connection[1][1]
            in_node_connection_variable = connection[1][0]


            future = out_node.connect_distributed(out_node_connection_variable, in_node_connection_variable, in_node)
            futures.append(future)
            self.connections.append(connection)
        
        self.get_results(futures)
        
    def swap_node(self, node_name, parameters):
        parameters["identifier"] = self.nodes[node_name].get_parameter_value("identifier").result()
        
        node_type = self.available_classes[parameters["type"]]

        if node_name in self.nodes:
            self.nodes[node_name] = self.client.submit(node_type, parameters, actor = True).result()
        else:
            print("Node with the name: ", node_name, " not found")
        map_list = node_name.split("-")
        map_list = map_list[1:]
        map_list.append("parameters")
        set_node_value_in_genome(self.graph_genome, map_list, parameters)

    def set_node_static_state_variable(self, node_name, variable_name, new_value):
        self.nodes[node_name].set_static_state_variable(variable_name, new_value)

    
    def save_graph(self, folder_name = ""):
        try:
            if folder_name == "":
                folder_name = self.graph_genome["identifier"]
            else:
                self.graph_genome["identifier"] = folder_name
            os.makedirs(folder_name)
        except FileExistsError:
            print()
            folder_name = input("A graph with the same name has already been saved, please provide a different graph name\n New name: ")
            self.graph_genome["indentifier"] = folder_name
            os.makedirs(folder_name)
            #To do: Allow user to provide new name in terminal
        

        base_path = os.getcwd()
        file_name = os.path.join(base_path, folder_name, "graph_genome.json")
        with open(file_name, "w") as fp:
            json.dump(self.graph_genome, fp, sort_keys = False, indent = 4)

        
        for identifier, node in self.nodes.items():
            state = node.get_state().result()
            for state_type in state:
                file_name = identifier + "-" + state_type
                for state_name in state[state_type]:
                    file_name = file_name + "-" + state_name + ".npy"
                    file_name = os.path.join(base_path, folder_name, file_name)
                    np.save(file_name, state[state_type][state_name])

    def load_graph(self, folder_name, swappable_nodes = False):
        base_path = os.getcwd()
        file_name = os.path.join(base_path, folder_name, "graph_genome.json")
        with open(file_name) as json_file:
            self.graph_genome = json.load(json_file)
            

        self.construct_nodes(self.graph_genome) 

        futures = []
        for identifier, node in self.nodes.items():
            state = node.get_state().result()
            loaded_state = {}
            for state_type in state:
                file_name = identifier + "-" + state_type
                loaded_state[state_type] = {}
                
                for state_name in state[state_type]:
                    file_name = file_name + "-" + state_name + ".npy"
                    file_name = os.path.join(base_path, folder_name, file_name)
                    loaded_state[state_type][state_name] = np.load(file_name)

            future = node.set_state(loaded_state)
            futures.append(future)
        self.get_results(futures)

        if swappable_nodes == False:
            self.connect_nodes
        
    def increment(self):
        futures = []
        for identifier, node in self.nodes.items():
            future = node.compute_next()
            futures.append(future)
        self.get_results(futures)

        futures = []
        for identifier, node in self.nodes.items():
            future = node.update_current()
            futures.append(future)
        self.get_results(futures)
    
    
    def save_memories(self, folder_name):
        try:
            os.makedirs(folder_name)

        except FileExistsError:
            print()
            folder_name = input("A folder with the name: " + folder_name + " already exists" +"\"n New name: ")
            os.makedirs(folder_name)
            #To do: Allow user to provide new name in terminal
        
        for identifier, node in self.nodes.items():
            sub_folder_name = identifier.split('-')
            sub_folder_name = '/'.join(sub_folder_name)
            sub_folder_name = os.path.join(folder_name, sub_folder_name)

            memories = node.get_memories().result()

            if len(memories) > 0:
                os.makedirs(sub_folder_name)
                for variable_name, memory in memories.items():
                        file_name = os.path.join(sub_folder_name, variable_name)
                        np.save(file_name, memory)
        
    
             
    def get_results(self, futures_list):
        for i in range(len(futures_list)):
            futures_list[i].result()