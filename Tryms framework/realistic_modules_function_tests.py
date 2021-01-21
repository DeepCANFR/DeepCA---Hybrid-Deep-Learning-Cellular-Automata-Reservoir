# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:09:25 2021

@author: trymlind
"""


import realistic_module_test as rm
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

if True:
    
    #
    nr_of_tests = 100
    delay_upper_bound = 100
    error_tolerance = 1
    
 
    delay_line_10ms_dt_1_parameteres = {}
    
    delay_line_10ms_dt_1_parameteres["delay"] = 3 #ms
    delay_line_10ms_dt_1_parameteres["time_step"] = 0.1
    
    delay_line_10ms_dt_1 = rm.Delay_Line(delay_line_10ms_dt_1_parameteres)
    
    dt = delay_line_10ms_dt_1_parameteres["time_step"]
    
    simulation_time = 100 #ms
    simulation_time_steps = simulation_time / dt
    time_start = 0
    time_stop = 1
    
    input_times = cp.array([10,13,20,21,22,23,27])
    excpected_output_times = input_times + delay
    
    t = 0
    input_nr = 0
    iutput_nr = 0
    
    current_input = cp.array([0])
    delay_line_10ms_dt_1.interface_read_variable(current_input)
    
    correct_timing = 0
    for time_step in simulation_time_steps:
        if t == expected_output_times[input_nr]:
            corrent_timing += delay_line_10ms_dt_1.current_spike_output == 1
        else:
            corrent_timing += delay_line_10ms_dt_1.current_spike_output == 0
        
        if t == input_times[input_nr]:
            current_input[:] = input_times[current_input]
        else:
            current_input[:] = 0
        
        delay_line_10ms_dt_1.compute_new_values()
        delay_line_10ms_dt_1.update_current_values()
        
        t += dt
        
        

    
    