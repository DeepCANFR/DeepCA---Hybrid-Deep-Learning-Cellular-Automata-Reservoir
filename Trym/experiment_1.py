# import numpy as np

# from dask.distributed import Client
from dask_jobqueue import SLURMCluster
import time
import numpy as np
from dask.distributed import Client

class Node:
    def __init__ (self, parameters):
        time.sleep(1)
        #states must be numpy/cupy arrays
        self.current_state = {}
        self.new_state = {}
        self.static_state = {}
        self.memory = {}
        self.parameters = parameters
        self.connected_local_nodes = {}
        self.connected_distributed_nodes = {}
        self.t = 0
        
    def create_new_state_values(self):
        for key in self.current_state:
            self.new_state[key] = np.copy(self.current_state[key])
        
    def compute_new(self):
        pass
    
    def update_current(self):
        self.update_current_local()
        self.update_current_distributed()
        self.update_current_internal()

        for key in self.memory:
            self.memory[key]
            # ToDo: Figure out how to store memories in a generic way
            '''
            The problem is that it would be best to store the memory in a numpy array. 
            However the current state should be inserted into this array in a generic manner.
            The problem being that the state array could be 2d or 3d or 4d, so how can I create a 
            way to index into the right axis in a generic manner
            '''

    def update_current_internal(self):
        '''
        This function updates the internal values with corresponding internal new values
        Since new values 
        '''
        for key in self.new_state:
            new_values = self.new_state[key]
            current_values = self.current_state[key]
            np.copyto(current_values, new_values)
    
    def update_current_local(self):
        '''
        Update local is used for nodes running in a serial fashion, instead of in parallel and distributed
        '''
        for node_identifier in self.connected_local_nodes:
            external_node = self.connected_local_nodes[node_identifier]["node"]
            internal_value_name = self.connected_local_nodes[node_identifier]["internal_value_name"]
            external_value_name = self.connected_local_nodes[node_identifier]["external_value_name"]
            
            internal_value = self.current_state[internal_value_name]
            external_value = self.connected_local_nodes[node_identifier]["node"].get_new_state_value(external_value_name)
            
            np.copyto(internal_value, external_value)
        
        
        
    
    async def update_current_distributed(self):
        '''
        Because workers may only have one thread we cannot necessarily call another node on the same
        worker in parallel. This may lead to deadlocks or other bugs. For this reason the nodes that
        have been submitted to run in parallel through the client (but may live on the same worker) 
        need to be called with an async function. Furthermore call need to use await when getting the external value
        Note that the .get_new_state_value() method also uses async in its definition
        '''
        for node_identifier in self.connected_distributed_nodes:
            external_node = self.connected_distributed_nodes[node_identifier]["node"]
            internal_value_name = self.connected_distributed_nodes[node_identifier]["internal_value_name"]
            external_value_name = self.connected_distributed_nodes[node_identifier]["external_value_name"]
            
            internal_value = self.current_state[internal_value_name]
            external_value = await self.connected_distributed_nodes[node_identifier]["node"].get_new_state_value(external_value_name)
            # ToDo: Consider adapting external value call so that it does not need to call result() but instead get futures
            # ToDo: Consider allowing the get_value() call to return multiple values in case more than one is needed from the same node
            
            np.copyto(internal_value, external_value)


        for key in self.memory:
            self.memory[key]
            # ToDo: Figure out how to store memories in a generic way
            '''
            The problem is that it would be best to store the memory in a numpy array. 
            However the current state should be inserted into this array in a generic manner.
            The problem being that the state array could be 2d or 3d or 4d, so how can I create a 
            way to index into the right axis in a generic manner
            '''
            
        
    def connect_local(self, internal_value_name, external_value_name, node, node_identifier):
        '''
        connect_local is used to connect nodes that are not run in a distributed fashion but instead
        run serially. Since the update call is different between local and dirstributed nodes
        they need to be separated which is the motivation for having two different methods for connecting
        nodes
        '''
        self.connected_local_nodes[node_identifier] = {
            "node":node,
            "external_value_name": external_value_name,
            "internal_value_name": internal_value_name,
        }
    def connect_distributed(self, internal_value_name, external_value_name, node, node_identifier):
        '''
        see connect_local
        '''
        self.connected_distributed_nodes[node_identifier] = {
            "node":node,
            "external_value_name": external_value_name,
            "internal_value_name": internal_value_name,
        }
        
    
    async def get_current_state_value(self, value_name):
        return self.current_state[value_name]
    
    async def get_new_state_value(self, value_name):
        return self.new_state[value_name]

    def save_state(self):


class TestNode(Node):
    def __init__(self, parameters):
        super().__init__(parameters)
        a_size = self.parameters["a_size"]
        self.current_state["a"] = np.zeros(a_size)
        self.create_new_state_values()
        
    def compute_new(self):
        time.sleep(1)
        current_value = self.current_state["a"]
        new_value = self.new_state["a"]
        
        np.copyto(new_value, current_value*2)
        
    
    def set_current_value(self, value):
        current_value = self.current_state["a"]
        
        np.copyto(current_value, value)
        time.sleep(1)




def get_results(futures_list):
    for i in range(len(futures_list)):
        futures_list[i].result()
        
graph_length = 12
sim_length = 10

current_values = np.zeros(graph_length)
new_values = np.zeros(graph_length)

if __name__ == "__main__":
    # with SLURMCluster(cores=6,
    #                         processes=4,
    #                         memory="1024GB",
    #                         project="dashproject",
    #                         walltime="01:00:00",
    #                         queue="defq") as cluster:

    

    with Client(n_workers = graph_length) as client:

        print("submitting nodes")
        graph = []
        for i in range(graph_length):
            parameters = {
                "identifier":i,
                "a_size":1
            }
            graph.append(client.submit(TestNode, parameters, actor = True))
        print("getting node proxies")
        for i in range(graph_length):
            graph[i] = graph[i].result()
        
        futures = []
        print("connecting nodes")
        for i in range(graph_length):
            futures.append(graph[i].connect_distributed("a", "a", graph[i-1], i-1))
        print("getting connection results")
        get_results(futures)
        
        print("setting current value of node 0 as 1")
        graph[0].set_current_value(1).result()
        
        
        print("starting simulation")
        for t in range(sim_length):
            t0 = time.time()
            print(t)
            
            print("updating current values")
            futures = []
            for i in range(graph_length):
                print(graph[i].get_current_state_value("a").result())
                
            for i in range(graph_length):
                futures.append(graph[i].compute_new())
            get_results(futures)
            
            print()
            print("computing new values")
            for i in range(graph_length):
                print(graph[i].get_new_state_value("a").result())
            
            
            futures = []
            for i in range(graph_length):
                futures.append(graph[i].update_current_distributed())
            get_results(futures)
            print(time.time() - t0)
            
            