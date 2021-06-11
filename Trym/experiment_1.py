# import numpy as np

# from dask.distributed import Client

import time
import numpy as np
from dask.distributed import Client
from copy import deepcopy
import json
import os
from functools import reduce 
import operator
import sys

class Node:
    def __init__ (self, parameters):
        time.sleep(1)
        #states must be numpy/cupy arrays
        self.current_state = {} # The current state is used to compute the next state
        self.next_state = {} # next state is the future state of the computations the node performs
        self.static_state = {} # static state contain variable that never change
        self.memories = {} # memories contains the values we wish to store and use for analysis later
        self.parameters = parameters # parameters are values that are used once to initialize the variables in current, next and static state. It also includes information on what to store memories on
        self.connected_local_nodes = {}
        self.connected_distributed_nodes = {}
        self.t = 0

        
        if "memories" in self.parameters:
            for key in self.parameters["memories"]:
                self.memories[key] = []
        
        
    def copy_next_state_from_current_state(self):
        for key in self.current_state:
            self.next_state[key] = np.copy(self.current_state[key])
        
    def compute_next(self):
        pass
    
    async def update_current(self):
        self.update_current_local()
        await self.update_current_distributed()
        self.update_current_internal()

        for key in self.memories:
            state_copy = np.copy(self.current_state[key])
            self.memories[key].append(state_copy)
            # ToDo: Figure out how to store memories in a generic way
            '''
            The problem is that it would be best to store the memories in a numpy array. 
            However the current state should be inserted into this array in a generic manner.
            The problem being that the state array could be 2d or 3d or 4d, so how can I create a 
            way to index into the right axis in a generic manner
            '''

    def update_current_internal(self):
        '''
        This function updates the internal values with corresponding internal next values
        Since next values 
        '''
        for key in self.next_state:
            next_values = self.next_state[key]
            current_values = self.current_state[key]
            np.copyto(current_values, next_values)
    
    def update_current_local(self):
        '''
        Update local is used for nodes running in a serial fashion, instead of in parallel and distributed
        '''
        for node_identifier in self.connected_local_nodes:
            external_node = self.connected_local_nodes[node_identifier]["node"]
            internal_value_name = self.connected_local_nodes[node_identifier]["internal_value"]
            external_value_name = self.connected_local_nodes[node_identifier]["external_value"]
            
            internal_value = self.current_state[internal_value_name]
            external_value = self.connected_local_nodes[node_identifier]["node"].get_next_state_value(external_value_name)
            
            np.copyto(internal_value, external_value)
        
    async def update_current_distributed(self):
        '''
        Because workers may only have one thread we cannot necessarily call another node on the same
        worker in parallel. This may lead to deadlocks or other bugs. For this reason the nodes that
        have been submitted to run in parallel through the client (but may live on the same worker) 
        need to be called with an async function. Furthermore calls need to use await when getting the external value
        Note that the .get_next_state_value() method also uses async in its definition
        '''
        for node_identifier in self.connected_distributed_nodes:
            external_node = self.connected_distributed_nodes[node_identifier]["node"]
            internal_value_name = self.connected_distributed_nodes[node_identifier]["internal_value"]
            external_value_name = self.connected_distributed_nodes[node_identifier]["external_value"]
            
            internal_value = self.current_state[internal_value_name]
            external_value = await external_node.get_next_state_value(external_value_name)
            # ToDo: Consider adapting external value call so that it does not need to call result() but instead get futures
            # ToDo: Consider allowing the get_value() call to return multiple values in case more than one is needed from the same node
            
            np.copyto(internal_value, external_value)

            
        
    async def connect_local(self, internal_value_name, external_value_name, node):
        '''
        connect_local is used to connect nodes that are not run in a distributed fashion but instead
        run serially. Since the update call is different between local and dirstributed nodes
        they need to be separated which is the motivation for having two different methods for connecting
        nodes
        '''
        node_identifier = await node.get_parameter_value("identifier")
        self.connected_local_nodes[node_identifier] = {
            "node":node,
            "internal_value": internal_value_name,
            "external_value": external_value_name
        }
        
    


    async def connect_distributed(self, internal_value_name, external_value_name, node):
        '''
        see connect_local
        '''
        node_identifier = await node.get_parameter_value("identifier")
        self.connected_distributed_nodes[node_identifier] = {
            "node":node,
            "internal_value": internal_value_name,
            "external_value": external_value_name
        }

    def set_current_value(self, value_name, value):
        current_value = self.current_state[value_name]
        
        np.copyto(current_value, value)
        
    
    async def get_current_state_value(self, value_name):
        return self.current_state[value_name]
    
    async def get_next_state_value(self, value_name):
        return self.next_state[value_name]
    
    async def get_parameter_value(self, value_name):
        return self.parameters[value_name]
    
    def get_memories(self):
        for key in self.memories:
            self.memories[key] = np.array(self.memories[key])
        return self.memories

    def get_state(self):
        state = {
            "current_state":self.current_state,
            "next_state":self.next_state,
            "static_state":self.static_state
        }
        return state
    
    def set_state(self, state):
        self.current_state = state["current_state"] 
        self.next_state = state["next_state"]
        self.static_state = state["static_state"]

    def set_static_state_variable(self, variable_name, new_value):
        self.static_state[variable_name] = new_value
        

    def get_results(self, futures_list):
        for i in range(len(futures_list)):
            futures_list[i].result()



def copy_graph_genome(graph_genome, new_name):
    graph_copy = deepcopy(graph_genome)
    graph_copy["identifier"] = new_name
    return graph_copy


def embed_sub_graph(super_graph, sub_graph):
    pass


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
            out_node_identifier = connection[0][0]
            in_node_identfier = connection[0][1]
            
            out_node = self.nodes[out_node_identifier]
            in_node = self.nodes[in_node_identfier]

            out_node_connection_variable = connection[1][0]
            in_node_connection_variable = connection[1][1]


            future = out_node.connect_distributed(out_node_connection_variable, in_node_connection_variable, in_node)
            futures.append(future)
            self.connections.append(connection)
        
        self.get_results(futures)
        
    def swap_node(self, node_name, parameters):
        node_type = self.available_classes[parameters["type"]]
        self.nodes[node_name] = self.client.submit(node_type, parameters, actor = True).result()
        map_list = node_name.split("-")
        map_list.append("parameters")
        set_node_value_in_genome(self.graph_genome, map_list, parameters)

    def set_node_static_state_variable(self, node_name, variable_name, new_value):
        self.nodes[node_name].set_static_state_variable(variable_name, name_value)

    
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

            '''
            for folder_name in create_folder_tree_names_from_genome(self.graph_genome, folder_name):
                print(folder_name)
                os.makedirs(folder_name)
                node_identifier = folder_name.split('/')
                node_identifier = node_identifier[1:]
                node_identifier = '-'.join(node_identifier)
                memories = self.nodes[node_identifier].get_memories().result()
                for variable_name, memory in memories.items():
                    file_name = os.path.join(folder_name, variable_name)
                    np.save(file_name, memory)
            '''

                
            
        except FileExistsError:
            print()
            folder_name = input("A folder with the name  name\n New name: ")
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

                    


class TestNode(Node):
    '''
    This class should demonstrate how to create subclasses of node
    '''
    def __init__(self, parameters):
        super().__init__(parameters)
        a_size = self.parameters["a_size"]
        b_size = self.parameters["b_size"]
        c_size = self.parameters["c_size"]

        # First create internal variables that has next_state and call copy_next_state_from_current_state to create arrays of same size and type in next_state
        self.current_state["a"] = np.zeros(a_size)
        self.current_state["b"] = np.zeros(b_size)
        self.copy_next_state_from_current_state() # maybe a poor implementation? Will fuck up update_internal() if used in the wrong order

        # Then create current values for those variables that the node does not itself compute next_states for
        self.current_state["c"] = np.zeros(c_size)

        # Lastly create static values
        upper_m = self.parameters["m"]["upper_bound"]
        lower_m = self.parameters["m"]["lower_bound"]
        self.static_state["m"] = np.random.uniform(lower_m, upper_m,1)
        
    def compute_next(self):
        #time.sleep(1)
        current_a = self.current_state["a"]
        current_b = self.current_state["b"]
        current_c = self.current_state["c"]

        next_a = self.next_state["a"]
        next_b = self.next_state["b"]

        static_m = self.static_state["m"]
        
        np.copyto(next_a, (current_a - current_b) + current_c * static_m)
        np.copyto(next_b, (current_c + current_b) + current_a * static_m)
        


class TestNodeDelay(Node):
    '''
    This class should demonstrate how to create subclasses of node
    '''
    def __init__(self, parameters):
        super().__init__(parameters)
        a_size = self.parameters["a_size"]
        b_size = self.parameters["b"]["size"]
        b_delay = self.parameters["b"]["delay"]
        c_size = self.parameters["c_size"]

        # First create internal variables and call copy_next_state_from_current_state to create arrays of same size and type in next_state
        self.current_state["a"] = np.zeros(a_size)
        self.copy_next_state_from_current_state() # maybe a poor implementation? Will fuck up update_internal() if used in the wrong order

        # Create internal variables that do not require a next state
        self.current_state["b"] = np.zeros((b_size, b_delay))

        # Then create current values for those variables that the node does not itself compute next_states for
        self.current_state["c"] = np.zeros(c_size)

        # Lastly create static values
        upper_m = self.parameters["m"]["upper_bound"]
        lower_m = self.parameters["m"]["lower_bound"]
        self.static_state["m"] = np.random.uniform(lower_m, upper_m,1)
        
    def compute_next(self):
        #time.sleep(1)
        c_a = self.current_state["a"]
        c_b = self.current_state["b"]
        c_c = self.current_state["c"]

        n_a = self.next_state["a"]

        s_m = self.static_state["m"]
        
        
        np.copyto(n_a, (c_a - c_b[:,-1]) + c_c * s_m)
        n_b = np.roll(c_b, 1, axis = 1)
        n_b[:,0] = c_a
        np.copyto(c_b, n_b)

        
   




if __name__ == "__main__":
    '''
    Running a simulation
    Step 1: Create graph genome.
    The first step is to create the parameters and structures of the graph in the form of a "genome". 
    Below this is done for a graph with multiple sub-graphs that are essentiall idenitcal to each other 
    which creates a natural hierachy of graphs
    At the lowest level we have to define nodes and the connections between them
    We do this by creating a dictionary which needs to contains certain elements:

    "identifier"    - is the name of this sub-graph
    "nodes"         - is a nested dictionary where the keys are the names of the nodes
                        - the sub directory must contain:
                            - "identifier"  - the name of the node
                            - "parameters"  - the parameters that serve as input to the node
                                              these will vary depending on the type of nodes you use
                                              Note that if you name state variables in the memory list
                                              these will be stored by the node and can later be saved 
                                              to disk
    "connections"   - is a nested list:
                        - the first level contains connections and has as many elements as there are connections
                            - the second level has two elements
                                - the first contains a list with two elements, 
                                    - the first element is the identifier of the out-node (using graph notation of directed edges)
                                    - the second is the identifier of the in-node
                                - the second element is another list with two elements
                                    - the first is the state variable in the out-node that is transfered to the out-node
                                    - the second is the state variable in the in-node that receives the state variable from the out-node
    
    '''
    sub_graph_0 = {
        "identifier":"sub_graph_0",
        "nodes": {
            "node_0":{
                "identifier":"node_0",
                "parameters":{
                    "type":"TestNode",
                    "a_size":1,
                    "b_size":1,
                    "c_size":1,
                    "m":{
                        "lower_bound":0,
                        "upper_bound":1
                    },
                    "memories":["a"]
                }
            },
            "node_1":{
                "identifier":"node_1",
                "parameters":{
                    "type":"TestNode",
                    "a_size":1,
                    "b_size":1,
                    "c_size":1,
                    "m":{
                        "lower_bound":0,
                        "upper_bound":1
                    },
                    "memories":[]
                }
            },
            "node_2":{
                "identifier":"node_2",
                "parameters":{
                    "type":"TestNode",
                    "a_size":1,
                    "b_size":1,
                    "c_size":1,
                    "m":{
                        "lower_bound":0,
                        "upper_bound":1
                    },
                    "memories":["c"]
                }
            }
        },
        "connections":[                 # connection is a nested list that described connections between nodes
            [
                ["node_0","node_1"],    # first element: out node, second element: in node
                ["c","a"]               # first element: variable in out node, second element: variable in in node. These are the variable that form the connection
            ],
            [
                ["node_1","node_2"],
                ["c","a"]
            ]
        ] 
    }    

    '''
    Because many graph types will have repeating sub-graphs we have the function copy_graph_genome()
    This allows you to create a copy of a sub-graph which can be renames and then connected to other sub-graphs
    '''
    sub_graph_1 = copy_graph_genome(sub_graph_0, "sub_graph_1")
    sub_graph_2 = copy_graph_genome(sub_graph_0, "sub_graph_2")

    '''
    When we create a super graph all we need to do is to create a dict that contains 
    the identifier of this super graph, the sub-graphs that comprises it and the connections between them
    The difference between a graph at the lowest level and a super-graph is that instead of containing node parameters
    it actually contains sub-graph genomes.
    Another key difference is that we now need to specifiy in which sub-graph the nodes that form the connection is located
    This is done by adding the identifier of the sub-graph to the identifier of the identifier of the node with a - in between
    '''
    super_graph_0 = {
        "identifier":"super_graph_0",
        "graphs":{
            "sub_graph_0":sub_graph_0,
            "sub_graph_1":sub_graph_1,
            "sub_graph_2":sub_graph_2
        },
        "connections":[
            [
                ["sub_graph_0-node_2","sub_graph_1-node_0"],
                ["c","a"]
            ],[
                ["sub_graph_1-node_2","sub_graph_2-node_0"],
                ["c","a"]
            ],[
                ["sub_graph_2-node_2","sub_graph_0-node_0"],
                ["c","a"]
            ]  
        ]
    }

    '''
    Copying can also be done on super-graphs in the same way as sub-graphs
    '''
    super_graph_1 = copy_graph_genome(super_graph_0, "super_graph_1")

    '''
    We can also create super-super-graphs which itself contains another level of sub-graphs
    This is as we did with the first level of sub-graphs. 
    This time we need to add both the super-super graph identifier and the super-graph identifier to
    the node identifier to find be able to find them in the graph. 
    This can be done to an arbitrary level in the graph hiearchy as all you need to do is to add
    another level of identifiers so that it is possible to find the node from the perspective of the current graph level
    '''
    super_duper_graph = {
        "identifier":"super_duper_graph",
        "graphs":{
            "super_graph_0":super_graph_0,
            "super_graph_1":super_graph_1,
            "sub_graph_0":sub_graph_0
        },
        "connections":[
            [
                ["super_graph_0-sub_graph_2-node_2", "super_graph_1-sub_graph_0-node_0"],
                ["c","a"]
            ],[
                ["super_graph_1-sub_graph_2-node_2", "super_graph_0-sub_graph_0-node_0"],
                ["c","a"]  
            ],[
                ["super_graph_1-sub_graph_2-node_2", "sub_graph_0-node_0"],
                ["c", "a"]
            ]
        ]
    }

    
    sim_length = 10

    with Client(n_workers = 12) as client:
        '''
        When we actually build a graph we first need to initialize an instance of the class graph. If creating a fully distributed graph
        we use the DistributedGraph class which will send every node to a separate worker if enough are available
        The graph is initialized simply by giving it the client used by dask to schedule workers
        '''
        available_classes = {
            "TestNode":TestNode,
            "TestNodeDelay":TestNodeDelay
        }
        graph = DistributedGraph(client, available_classes)

        '''
        The next step is to create the graph itself which we do by calling construct_distributed_graph with the genome as input
        '''
        graph.construct_distributed_graph(super_duper_graph)
        '''
        We can also save the graph in its current state. using .save_graph()
        This will create a folder with the name of the highest level identifier in the graph genome if no input is given
        or a folder with a given string used as input
        Saving a graph is usefull if some state variables are initialized with some random factor described in the node parameters
        and you need to remove any noise produced by different initiallizations of the random variables when testing the effect of 
        swapping nodes.
        Note that saving the graph will save it in its current state, so if you want to save the graph at after running some arbitrary
        amount of time steps this is possible
        '''
        graph.save_graph()
        '''
        To incremenet the graph forward one time step we call increment. And to run a simulation we do this inside a for loop

        '''

        new_node_parameters = {
            "type":"TestNodeDelay",
            "a_size":1,
            "b":{
                "size":1,
                "delay":3
            },
            "c_size":1,
            "m":{
                "lower_bound":0,
                "upper_bound":1
            },
            "memories":["c"]
        }
        graph_2 = DistributedGraph(client, available_classes)
        graph_2.load_graph("super_duper_graph")
        graph_2.construct_nodes()
        graph_2.swap_node("super_graph_0-sub_graph_1-node_0", new_node_parameters)
        graph_2.connect_nodes()
        graph_2.save_graph("super_duper_graph_modified")

        for t in range(sim_length):
            graph.increment()
            graph_2.increment()
        
        graph.save_memories("simulation_1")
        graph_2.save_memories("modified_simulation_1")
        '''
        If your node parameters contain any variable names under the memories key you can now have these returned to the main script and saved 
        for later analysis using .save_memories() This will create a new folder with the data
        '''
        

