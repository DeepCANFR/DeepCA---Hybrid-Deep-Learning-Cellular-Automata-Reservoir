# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:04 2021

@author: trymlind
"""


import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
if False:
    sim_time = 1000
    population_size = (1000,1000)
    delta_t = 1
    time_since_last_spike = cp.zeros(population_size)
    
    spikes = cp.random.rand(population_size[0], population_size[1], sim_time) < 0.1
    
    start_time = time.time()
    for t in range(sim_time):
        time_since_last_spike += delta_t
        time_since_last_spike *= spikes[:,:,t] == 0
    print("*= inverse spikes: ", time.time() - start_time)
    
    time_since_last_spike = cp.zeros(population_size)
    start_time = time.time()
    for t in range(sim_time):
        time_since_last_spike += delta_t
        spike_indexes = cp.where(spikes[:,:,t] == 0)
        time_since_last_spike[spike_indexes[0], spike_indexes[1]] = 0
    print("cp.where: ", time.time() - start_time)
    

if False:
    x = cp.array([1000000.0])
    while x <= 1000000:
        x *= 0.1
        print(x)
        
if False:
    class Exponential_decay(V,t):
        def __init__(self):
            self.last_max_V = 0
            self.current_V
            self.new_V
            self.t_since_spike = 0
            self.time_step
        def compute_new_value(inputs):
            self.last_max_V += inputs
            if inputs > 0:
                self.time_since_last_spike = 0
            else:
                self.time_since_last_spike += self.time_step
            
if True:
    resting_utilization_of_synaptic_efficacy = -2
    current_utilization_of_synaptic_efficacy = 1.2
    new_utilization_of_synaptic_efficacy = 1.3
    current_neurotransmitter_reserve = 1.2
    tau_facil = 30
    tau_recovery = 30
    
    time_since_last_spike = 0
    
    sim_t = 100
    r = np.zeros(sim_t)
    u = np.zeros(sim_t)
    t = np.zeros(sim_t)
    for i in range(sim_t):
        
        new_utilization_of_synaptic_efficacy = current_utilization_of_synaptic_efficacy * cp.exp((-time_since_last_spike) / tau_facil) + resting_utilization_of_synaptic_efficacy*(1 - current_utilization_of_synaptic_efficacy * cp.exp((-time_since_last_spike) / tau_facil))
        new_neurotransmitter_reserve = current_neurotransmitter_reserve * (1 - new_utilization_of_synaptic_efficacy)*cp.exp(-time_since_last_spike / tau_recovery) + 1 - cp.exp(-time_since_last_spike / tau_recovery)
        
        current_utilization_of_synaptic_efficacy = new_utilization_of_synaptic_efficacy
        current_neurotransmitter_reserve = new_neurotransmitter_reserve
        time_since_last_spike += 1
        r[i] = current_neurotransmitter_reserve
        u[i] = current_utilization_of_synaptic_efficacy
        t[i] = cp.exp((-time_since_last_spike) / tau_facil)
    plt.figure(1)
    plt.plot(r)
    plt.plot(u)
    plt.plot(t)