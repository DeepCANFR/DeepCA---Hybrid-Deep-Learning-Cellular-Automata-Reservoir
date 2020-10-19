# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 20:46:01 2020

@author: trymlind
"""
import numpy as np
import framework_module as fm
import cv2

x = 500
y = 500 
class Static_Synapse_Neuron:
    
    def __init__(self):
        '''
        All components of must be placed in a list in an order corresponding to information flow
        '''
   
        
        self.soma = fm.Soma_AS(0.2,0.5,0.5,x,y)
        weight_parameters = {}
        weight_parameters["type"] = "normal"
        weight_parameters["mean"] = 0.3
        weight_parameters["sd"] = 0.3
        self.postsynaptic_terminal = fm.Postsynaptic_terminal_weighted(np.zeros((x,y,8)), weight_parameters, lock = "1")
        self.components = [self.soma, self.postsynaptic_terminal]
        self.forward_output = np.full(self.components[0].shape,0)
        
    def transfer(self, inputs):
        #print('\n Static synapse transfer')
        self.forward_output = self.components[0].transfer(self.components[1].forward_output, self.components[1].backward_output)
        for i0 in range(1,len(self.components)-1):
            #print(i0)
            self.components[i0].transfer(self.components[i0+1].forward_output, self.components[i0+1].backward_output)
    
        
        self.components[-1].transfer(inputs, backward_inputs = None)
        
    def compute(self):
        for i0 in range(len(self.components)):
            self.components[i0].compute()
            
class Dynamic_Synapse_Neuron(Static_Synapse_Neuron):
    def __init__(self):
        super().__init__()
        inputs = np.zeros((x,y), dtype ='float64')
        #print(inputs)
        neighbourhood_template = np.ones((3,3))
        neighbourhood_template[1,1] = 0
        #neighbourhood_template[0,0] = 0
        self.preynaptic_terminal = fm.Preynaptic_terminal(inputs, neighbourhood_template = neighbourhood_template, max_neurotransmiter = 10, release_ratio = 0.1, cleft_re_uptake_rate = 0.1, synaptic_gap_leak_rate= 0, key = '1')
        self.components.append(self.preynaptic_terminal)
        



input_array = np.zeros((x,y))
input_array[125,125] = 1

pop = Dynamic_Synapse_Neuron()
#pop = Static_Synapse_Neuron()
death_length = 0

#cap = cv2.VideoCapture(0)
previous_sums = np.zeros(6)
while True:
    
    #ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #gray = np.array(gray)
    
    pop.compute()
    pop.transfer(pop.soma.spiked_neurons + input_array)
    
    input_array = np.zeros((x,y))
    if np.sum(previous_sums) > -1:
        death_length += 1
        print("fu")
        if death_length > -1:
            input_array[np.random.randint(125,175),np.random.randint(125,175)] = 2
            death_length = 0
            print("oyh")
    if np.sum(np.sum(pop.soma.spiked_neurons)) != 0:
         video = np.uint8(pop.soma.spiked_neurons)*255
         cv2.imshow('frame', video)  
    previous_sums = np.roll(previous_sums, 1)
    previous_sums[0] = np.sum(np.sum(pop.soma.spiked_neurons))
    print(previous_sums)
    
    #network.network_state[0,:] = 0

    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()