
import pickle

import parallel_realistic_module_V2 as rm
import matplotlib.pyplot as plt
#import tensorflow as tf
from scipy import stats
import dask
from dask.distributed import Client, LocalCluster
from dask.distributed import performance_report
from collections import OrderedDict

import sys

import numpy as np
import numpy as ncp
import cv2


weight_scaling = 1

time_step = 1
population_size = (10,100) #image_train[0,:,:].shape
upper_limit = 10000

'''
Somas
'''
# Excitatory (E)
E_soma_parameters = {}
E_soma_parameters["type"] = rm.Circuit_Equation_Integrate_and_Fire_Soma
E_soma_parameters["population_size"] = population_size
E_soma_parameters["membrane_time_constant"] = {"distribution":"homogenous", "value":30} # ms
E_soma_parameters["absolute_refractory_period"] = 3 # ms
E_soma_parameters["threshold"] = {"distribution":"homogenous", "value":15} # mv
E_soma_parameters["background_current"] = 9 # nA
E_soma_parameters["input_resistance"] = {"distribution":"homogenous", "value":1} # M_Ohm
E_soma_parameters["refractory_period"] = 3
E_soma_parameters["time_step"] = time_step
E_soma_parameters["temporal_upper_limit"] = upper_limit
E_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":9}# mv

# Inhibitory (I)
I_soma_parameters = {}
I_soma_parameters["type"] = rm.Circuit_Equation_Integrate_and_Fire_Soma
I_soma_parameters["population_size"] = population_size
I_soma_parameters["membrane_time_constant"] = {"distribution":"homogenous", "value":30} # ms
I_soma_parameters["absolute_refractory_period"] = 2 # ms
I_soma_parameters["threshold"] = {"distribution":"homogenous", "value":15} # mv
I_soma_parameters["background_current"] =9 # nA
I_soma_parameters["input_resistance"] = {"distribution":"homogenous", "value":1} # M_Ohm
I_soma_parameters["refractory_period"] = 2
I_soma_parameters["time_step"] = time_step
I_soma_parameters["temporal_upper_limit"] = upper_limit
I_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":9} # mv

unique_ID_dict_creator = rm.Unique_ID_Dict_Creator(30)

E_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
I_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)

neurons = {}
neurons["E1_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(E_1_soma_parameter_dict)
neurons["E2_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(E_2_soma_parameter_dict)
neurons["E3_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(E_3_soma_parameter_dict)

neurons["I1_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(I_1_soma_parameter_dict)
neurons["I2_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(I_2_soma_parameter_dict)
neurons["I3_soma"] = rm.Circuit_Equation_Integrate_and_Fire_Soma(I_3_soma_parameter_dict)



E1_dead_cells = ncp.load("experiment_0_layer_0_Excitatory_kill_mask.npy")
I1_dead_cells = ncp.load("experiment_0_layer_0_Inhibitory_kill_mask.npy")
E2_dead_cells = ncp.load("experiment_0_layer_1_Excitatory_kill_mask.npy")
I2_dead_cells = ncp.load("experiment_0_layer_1_Inhibitory_kill_mask.npy")
E3_dead_cells = ncp.load("experiment_0_layer_2_Excitatory_kill_mask.npy")
I3_dead_cells = ncp.load("experiment_0_layer_2_Inhibitory_kill_mask.npy")

neurons["E1_soma"].set_dead_cells(E1_dead_cells)
neurons["E2_soma"].set_dead_cells(E2_dead_cells)
neurons["E3_soma"].set_dead_cells(E3_dead_cells)

neurons["I1_soma"].set_dead_cells(I1_dead_cells)
neurons["I2_soma"].set_dead_cells(I2_dead_cells)
neurons["I3_soma"].set_dead_cells(I3_dead_cells)

somas = {}
for key in neurons:
    somas[key] = neurons[key].compile_data()
import cloudpickle
with open('homogenous_Circuit_equation_6_reset_treshold_distance_somas_1.pkl', 'wb') as output:
    cloudpickle.dump(somas, output)
