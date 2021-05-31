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
        self.state = {}
        self.parameters = parameters
        self.connected_nodes = {}
        
        
    def compute_new(self):
        pass
    
    def update_current(self):
        for node_ID in self.connected_nodes:
            external_node = self.connected_nodes[node_ID]["node"]
            internal_value_name = self.connected_nodes[node_ID]["internal_value_name"]
            external_value_name = self.connected_nodes[node_ID]["external_value_name"]
            
            internal_value = self.state[internal_value_name]
            external_value = self.connected_nodes[node_ID]["node"].get_state_value("external_value_name")
            
            np.copyto(internal_value, external_value)
            
    
    async def update_current_distributed(self):
        for node_ID in self.connected_nodes:
            external_node = self.connected_nodes[node_ID]["node"]
            internal_value_name = self.connected_nodes[node_ID]["internal_value_name"]
            external_value_name = self.connected_nodes[node_ID]["external_value_name"]
            
            internal_value = self.state[internal_value_name]
            external_value = await self.connected_nodes[node_ID]["node"].get_state_value(external_value_name)
            # ToDo: Consider adapting external value call so that it does not need to call result() but instead get futures
            # ToDo: Consider allowing the get_value() call to return multiple values in case more than one is needed from the same node
            
            np.copyto(internal_value, external_value)
            
        
    def connect(self, internal_value_name, external_value_name, node, node_ID):
        self.connected_nodes[node_ID] = {
            "node":node,
            "external_value_name": external_value_name,
            "internal_value_name": internal_value_name,
        }
        
    
    async def get_state_value(self, value_name):
        return self.state[value_name]
    

class TestNode(Node):
    def __init__(self, parameters):
        super().__init__(parameters)
        self.state["current_value"] = np.zeros(1)
        self.state["new_value"] = np.zeros(1)
        
    def compute_new(self):
        time.sleep(1)
        current_value = self.state["current_value"]
        new_value = self.state["new_value"]
        
        np.copyto(new_value, current_value*2)
        
    
    def set_current_value(self, value):
        current_value = self.state["current_value"]
        
        np.copyto(current_value, value)
        time.sleep(1)


def get_results(futures_list):
    for i in range(len(futures_list)):
        futures_list[i].result()
        
graph_length = 120
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
            graph.append(client.submit(TestNode, {"ID":i}, actor = True))
        print("getting node proxies")
        for i in range(graph_length):
            graph[i] = graph[i].result()
        
        futures = []
        print("connecting nodes")
        for i in range(graph_length):
            futures.append(graph[i].connect("current_value", "new_value", graph[i-1], i-1))
        print("getting connection results")
        get_results(futures)
        
        print("setting current value of node 0 as 1")
        graph[0].set_current_value(1).result()
        
        
        print("starting simulation")
        for t in range(sim_length):
            t0 = time.time()
            print(t)
            print("computing new values")
            futures = []
            for i in range(graph_length):
                print(graph[i].get_state_value("current_value").result())
                
            for i in range(graph_length):
                futures.append(graph[i].compute_new())
            get_results(futures)
            
            for i in range(graph_length):
                print(graph[i].get_state_value("new_value").result())
            
            print("updating current values")
            futures = []
            for i in range(graph_length):
                futures.append(graph[i].update_current_distributed())
            get_results(futures)
            print(time.time() - t0)
            
            