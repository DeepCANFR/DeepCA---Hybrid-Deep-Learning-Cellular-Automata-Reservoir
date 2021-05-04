
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
E_soma_parameters["membrane_time_constant"] = 15 # ms
E_soma_parameters["absolute_refractory_period"] = 3 # ms
E_soma_parameters["threshold"] = 15 # mv
#E_soma_parameters["reset_voltage"] = 13.5 # mv
E_soma_parameters["background_current"] = 13.5 # nA
E_soma_parameters["input_resistance"] = 1 # M_Ohm (maybe heterogenous)
E_soma_parameters["refractory_period"] = 3
E_soma_parameters["time_step"] = time_step
E_soma_parameters["temporal_upper_limit"] = upper_limit
#E_soma_parameters["membrane_recovery"] = 0.02 # a in model, represents K+ currents and Na+ inactivation. Time scale of recovery variable u
#E_soma_parameters["resting_potential_variable"] = 0.2 # b in model, sensitivity of recovery variable. Affects the resting potential, should be between -70 and -60 mV depending on the value of b
#E_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":-55}
E_soma_parameters["reset_voltage"] = {"distribution":"Izhikevich", "base_value":13.5, "multiplier_value": 15}# c in model,
#E_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":4} # d in model, after spike reset of the recovery variable u
#E_soma_parameters["reset_recovery_variable"] = {"distribution":"normal", "mean":8, "multiplier_value": -6}

# Inhibitory (I)
I_soma_parameters = {}
I_soma_parameters["type"] = rm.Circuit_Equation_Integrate_and_Fire_Soma
I_soma_parameters["population_size"] = population_size
I_soma_parameters["membrane_time_constant"] = 30 # ms
I_soma_parameters["absolute_refractory_period"] = 0 # ms
I_soma_parameters["threshold"] = 30 # mv
#I_soma_parameters["reset_voltage"] = 0#13.5 # mv
I_soma_parameters["background_current"] = 13.5 # nA
I_soma_parameters["input_resistance"] = 1 # M_Ohm
I_soma_parameters["refractory_period"] = 2
I_soma_parameters["time_step"] = time_step
I_soma_parameters["temporal_upper_limit"] = upper_limit
I_soma_parameters["membrane_recovery"] = {"distribution":"Izhikevich", "base_value":0.02, "multiplier_value": 0.08} # a in model, represents K+ currents and Na+ inactivation. Time scale of recovery variable u
I_soma_parameters["resting_potential_variable"] = {"distribution":"Izhikevich", "base_value":0.025, "multiplier_value": -0.05} # b in model, sensitivity of recovery variable. Affects the resting potential, should be between -70 and -60 mV depending on the value of b
I_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":-65} # c in model,
#I_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":0.2} # d in model, after spike reset of the recovery variable u
#I_soma_parameters["reset_voltage"] = {"distribution":"Izhikevich", "base_value":-65.8, "multiplier_value": 15}# c in model,
#E_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":4} # d in model, after spike reset of the recovery variable u
#I_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":-65}

unique_ID_dict_creator = rm.Unique_ID_Dict_Creator(30)

E_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
I_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)

neurons = {}
neurons["E_1_soma"] = rm.Izhikevich_Soma(E_1_soma_parameter_dict)
neurons["E_2_soma"] = rm.Izhikevich_Soma(E_2_soma_parameter_dict)
neurons["E_3_soma"] = rm.Izhikevich_Soma(E_3_soma_parameter_dict)

neurons["I_1_soma"] = rm.Izhikevich_Soma(I_1_soma_parameter_dict)
neurons["I_2_soma"] = rm.Izhikevich_Soma(I_2_soma_parameter_dict)
neurons["I_3_soma"] = rm.Izhikevich_Soma(I_3_soma_parameter_dict)



E1_dead_cells = ncp.load("experiment_0_layer_0_Excitatory_kill_mask.npy")
I1_dead_cells = ncp.load("experiment_0_layer_0_Inhibitory_kill_mask.npy")
E2_dead_cells = ncp.load("experiment_0_layer_1_Excitatory_kill_mask.npy")
I2_dead_cells = ncp.load("experiment_0_layer_1_Inhibitory_kill_mask.npy")
E3_dead_cells = ncp.load("experiment_0_layer_2_Excitatory_kill_mask.npy")
I3_dead_cells = ncp.load("experiment_0_layer_2_Inhibitory_kill_mask.npy")

neurons["E_1_soma"].set_dead_cells(E1_dead_cells)
neurons["E_2_soma"].set_dead_cells(E2_dead_cells)
neurons["E_3_soma"].set_dead_cells(E3_dead_cells)

neurons["I_1_soma"].set_dead_cells(I1_dead_cells)
neurons["I_2_soma"].set_dead_cells(I2_dead_cells)
neurons["I_3_soma"].set_dead_cells(I3_dead_cells)

import cloudpickle
with open('heterogenous_Circuit_equation_somas_1.pkl', 'wb') as output:
    cloudpickle.dump(neurons, output)
