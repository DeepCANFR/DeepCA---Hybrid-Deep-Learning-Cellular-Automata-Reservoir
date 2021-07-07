from dask.distributed import client
import numpy as np
import time



class Node:
    def __init__ (self, parameters):
        time.sleep(5)
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
        #print(self.current_state)

        for key in self.memories:
            if key in self.current_state:
                state_copy = np.copy(self.current_state[key])
                self.memories[key].append(state_copy)
            elif key in self.next_state: # saved some memory on stuff that only need a next state, but this desyncs the memories by one timestep
                state_copy = np.copy(self.next_state[key])
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
            if key in self.current_state:
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
            #print("\n\n\n", internal_value, "\n",external_value, "\n", self.current_state)
            del external_value
            
            

            
        
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
