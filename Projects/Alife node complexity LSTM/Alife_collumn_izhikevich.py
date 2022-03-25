# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 15:14:08 2021

@author: trymlind
"""
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


'''
Load stimuli
#######################################################################################
'''
#(image_train, label_train), (image_test, label_test) = tf.keras.datasets.mnist.load_data()


'''
Set parameters
######################################################################################
The first step is to setup the genereic parameters dictionaries. Since we are building
three layers (x2 one for inhibitor and one for excitatory) we will have many connections
That utilize the same type components with the same parameter settings. Since we are
using them for multiple components we do not give them an ID at this point other than
the object name which specifies which type of component the parameters are for
'''

weight_scaling = 0.1

time_step = 1
population_size = (10,100) #image_train[0,:,:].shape
upper_limit = 10000

'''
Somas
'''
# Excitatory (E)
E_soma_parameters = {}
E_soma_parameters["type"] = rm.Izhikevich_Soma
E_soma_parameters["population_size"] = population_size
E_soma_parameters["membrane_time_constant"] = 15 # ms
E_soma_parameters["absolute_refractory_period"] = 3 # ms
E_soma_parameters["threshold"] = 30 # mv
#E_soma_parameters["reset_voltage"] = 13.5 # mv
E_soma_parameters["background_current"] = 2#13.5 # nA
E_soma_parameters["input_resistance"] = 1 # M_Ohm
E_soma_parameters["refractory_period"] = 3
E_soma_parameters["time_step"] = time_step
E_soma_parameters["temporal_upper_limit"] = upper_limit
E_soma_parameters["membrane_recovery"] = 0.1 # a in model, represents K+ currents and Na+ inactivation. Time scale of recovery variable u
E_soma_parameters["resting_potential_variable"] = 0.26 # b in model, sensitivity of recovery variable. Affects the resting potential, should be between -70 and -60 mV depending on the value of b
E_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":-55} # c in model,
E_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":4} # d in model, after spike reset of the recovery variable u

# Inhibitory (I)
I_soma_parameters = {}
I_soma_parameters["type"] = rm.Izhikevich_Soma
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
I_soma_parameters["membrane_recovery"] = 0.02 # a in model, represents K+ currents and Na+ inactivation. Time scale of recovery variable u
I_soma_parameters["resting_potential_variable"] = 0.25 # b in model, sensitivity of recovery variable. Affects the resting potential, should be between -70 and -60 mV depending on the value of b
I_soma_parameters["reset_voltage"] = {"distribution":"homogenous", "value":-65} # c in model,
I_soma_parameters["reset_recovery_variable"] = {"distribution":"homogenous", "value":0.2} # d in model, after spike reset of the recovery variable u

input_parameters = {}
input_parameters["type"] = rm.Inputs_Distribute_Single_spike
input_parameters["population_size"] = population_size
input_parameters["percent"] = 1

'''
Dynamical synapses
'''
# Excitatory to Excitatory (EE)
EE_dynamical_synapse_parameters = {}
EE_dynamical_synapse_parameters["type"] = rm.Dynamical_Axonal_Terminal_Markram_etal_1998
EE_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U
EE_dynamical_synapse_parameters["time_constant_depression"] = {"distribution":"normal", "mean":1.1, "SD":1.1/2}# in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
EE_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
EE_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":30*weight_scaling, "SD":30*weight_scaling}# in Maas et al: A, in Markram et al: A
EE_dynamical_synapse_parameters["synapse_type"] = "excitatory"
EE_dynamical_synapse_parameters["time_step"] = time_step
EE_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

#Excitatory to Inhibitory (EI)
EI_dynamical_synapse_parameters = {}
EI_dynamical_synapse_parameters["type"] = rm.Dynamical_Axonal_Terminal_Markram_etal_1998
EI_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.5, "SD":0.5/2}# U  Strange, the setting from the paper is 0.05, but this results in the inhibitory neurons not firing
EI_dynamical_synapse_parameters["time_constant_depression"] = {"distribution":"normal", "mean":0.125, "SD":0.125/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
EI_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":1.2, "SD":1.2/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
EI_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":60*weight_scaling, "SD":60*weight_scaling}# in Maas et al: A, in Markram et al: A
EI_dynamical_synapse_parameters["synapse_type"] = "excitatory"
EI_dynamical_synapse_parameters["time_step"] = time_step
EI_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

# Inhibitory to Excitatory (IE)
IE_dynamical_synapse_parameters = {}
IE_dynamical_synapse_parameters["type"] = rm.Dynamical_Axonal_Terminal_Markram_etal_1998
IE_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.25, "SD":0.25/2} # U
IE_dynamical_synapse_parameters["time_constant_depression"] = {"distribution":"normal", "mean":0.7, "SD":0.7/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
IE_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.02, "SD":0.02/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
IE_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
IE_dynamical_synapse_parameters["synapse_type"] = "inhibitory"
IE_dynamical_synapse_parameters["time_step"] = time_step
IE_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit

# Inhibitory to Inhibitory (II)
II_dynamical_synapse_parameters = {}
II_dynamical_synapse_parameters["type"] = rm.Dynamical_Axonal_Terminal_Markram_etal_1998
II_dynamical_synapse_parameters["resting_utilization_of_synaptic_efficacy"] = {"distribution":"normal", "mean":0.32, "SD":0.32/2} # U
II_dynamical_synapse_parameters["time_constant_depression"] = {"distribution":"normal", "mean":0.144, "SD":0.144/2} # in Maas et al: D, in Markram et al: tau_depression  expressed in seconds
II_dynamical_synapse_parameters["time_constant_facilitation"] = {"distribution":"normal", "mean":0.06, "SD":0.06/2} # in Maas et al: F, in Markram et al: tau_facil
# to do: this should be from a gamma distribution I think, but I don't understand how they made it
II_dynamical_synapse_parameters["absolute_synaptic_efficacy"] = {"distribution":"normal", "mean":-19*weight_scaling, "SD":19*weight_scaling}# in Maas et al: A, in Markram et al: A
II_dynamical_synapse_parameters["synapse_type"] = "inhibitory"
II_dynamical_synapse_parameters["time_step"] = time_step
II_dynamical_synapse_parameters["temporal_upper_limit"] = upper_limit



'''
Delay lines (axons)
'''

EE_delay_line_parameters = {}
EE_delay_line_parameters["type"] = rm.Delay_Line
EE_delay_line_parameters["delay"] = 2#1.5 # ms
EE_delay_line_parameters["time_step"] = time_step #ms
EE_delay_line_parameters["temporal_upper_limit"] = upper_limit

EI_delay_line_parameters = {}
EI_delay_line_parameters["type"] = rm.Delay_Line
EI_delay_line_parameters["delay"] = 1#0.8
EI_delay_line_parameters["time_step"] = time_step
EI_delay_line_parameters["temporal_upper_limit"] = upper_limit

IE_delay_line_parameters = {}
IE_delay_line_parameters["type"] = rm.Delay_Line
IE_delay_line_parameters["delay"] = 1#0.8
IE_delay_line_parameters["time_step"] = time_step
IE_delay_line_parameters["temporal_upper_limit"] = upper_limit

II_delay_line_parameters = {}
II_delay_line_parameters["type"] = rm.Delay_Line
II_delay_line_parameters["delay"] = 1#0.8
II_delay_line_parameters["time_step"] = time_step
II_delay_line_parameters["temporal_upper_limit"] = upper_limit

'''
arbors
'''
EE_dendritic_arbor_parameters = {}
EE_dendritic_arbor_parameters["type"] = rm.Dendritic_Arbor
projection_template = ncp.ones((7,7))
projection_template[3,3] = 0
#projection_template[8,4:] = 1
#projection_template[2,0:2] = 1
#projection_template[1:-1,:] = 0
#projection_template[1,0] = 1
EE_dendritic_arbor_parameters["projection_template"] = projection_template
EE_dendritic_arbor_parameters["time_step"] = time_step
EE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.3, "lambda_parameter":5}
EE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
EE_dendritic_arbor_parameters["boundry_conditions"] = "closed"

EI_dendritic_arbor_parameters = {}
EI_dendritic_arbor_parameters["type"] = rm.Dendritic_Arbor
EI_dendritic_arbor_parameters["projection_template"] = ncp.ones((7,7))
EI_dendritic_arbor_parameters["projection_template"][3,3] = 0
#EI_dendritic_arbor_parameters["projection_template"][1:5,1:5] = 0
EI_dendritic_arbor_parameters["time_step"] = time_step
EI_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.2, "lambda_parameter":5}
EI_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
EI_dendritic_arbor_parameters["boundry_conditions"] = "closed"

IE_dendritic_arbor_parameters = {}
IE_dendritic_arbor_parameters["type"] = rm.Dendritic_Arbor
IE_dendritic_arbor_parameters["projection_template"] = ncp.ones((7,7))
IE_dendritic_arbor_parameters["projection_template"][3,3] = 0
IE_dendritic_arbor_parameters["time_step"] = time_step
IE_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}
IE_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
IE_dendritic_arbor_parameters["boundry_conditions"] = "closed"

II_dendritic_arbor_parameters = {}
II_dendritic_arbor_parameters["type"] = rm.Dendritic_Arbor
II_dendritic_arbor_parameters["projection_template"] = ncp.ones((7,7))
II_dendritic_arbor_parameters["projection_template"][3,3] = 0
II_dendritic_arbor_parameters["time_step"] = time_step
II_dendritic_arbor_parameters["distance_based_connection_probability"] = {"C":0.4, "lambda_parameter":5}
II_dendritic_arbor_parameters["temporal_upper_limit"] = upper_limit
II_dendritic_arbor_parameters["boundry_conditions"] = "closed"

'''
Dendritic spines
'''
#excitatory dendritic spine
E_dendritic_Spine_parameters = {}
E_dendritic_Spine_parameters["type"] = rm.Dendritic_Spine_Maas
E_dendritic_Spine_parameters["time_step"] = time_step
E_dendritic_Spine_parameters["time_constant"] = 3 # ms
E_dendritic_Spine_parameters["temporal_upper_limit"] = upper_limit

#inhibitory dendritic spine
I_dendritic_Spine_parameters = {}
I_dendritic_Spine_parameters["type"] = rm.Dendritic_Spine_Maas
I_dendritic_Spine_parameters["time_step"] = time_step
I_dendritic_Spine_parameters["time_constant"] = 6 # ms
I_dendritic_Spine_parameters["temporal_upper_limit"] = upper_limit


'''
Here we create the dictionaries that contain the parameter dicts for the sequence
of component a connection consist of. Note that we use the unique_ID_dict_creator object
to create a copy of the generic dict for a component that has a unique ID attached to it.
This ID is used when the connnection is built later.
'''
# use this class to create unique ID's for every parameter dict
unique_ID_dict_creator = rm.Unique_ID_Dict_Creator(30)

E_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
E_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(E_soma_parameters)
I_1_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_2_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)
I_3_soma_parameter_dict = unique_ID_dict_creator.create_unique_ID_dict(I_soma_parameters)

E_1_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
E_2_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
E_3_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
I_1_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
I_2_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
I_3_input_parameters = unique_ID_dict_creator.create_unique_ID_dict(input_parameters)
# Setup connection parameter dicts
# Excitatory self connections
self_E_1_parameter_dict = OrderedDict()
self_E_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
self_E_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
self_E_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
self_E_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

self_E_2_parameter_dict = OrderedDict()
self_E_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
self_E_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
self_E_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
self_E_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

self_E_3_parameter_dict = OrderedDict()
self_E_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
self_E_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
self_E_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
self_E_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

# Inhibitory self connections
self_I_1_parameter_dict = OrderedDict()
self_I_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
self_I_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
self_I_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
self_I_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

self_I_2_parameter_dict = OrderedDict()
self_I_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
self_I_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
self_I_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
self_I_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

self_I_3_parameter_dict = OrderedDict()
self_I_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
self_I_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
self_I_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
self_I_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer x to layer y
EE_1_2_parameter_dict = OrderedDict()
#EE_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_1_3_parameter_dict = OrderedDict()
EE_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_2_1_parameter_dict = OrderedDict()
EE_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_2_3_parameter_dict = OrderedDict()
EE_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer 3 to layer x
EE_3_1_parameter_dict = OrderedDict()
EE_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_3_2_parameter_dict = OrderedDict()
EE_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EE_delay_line_parameters)
EE_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dendritic_arbor_parameters)
EE_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)




# Excitatory to Excitatory connection from layer 1 to layer 2
II_1_2_parameter_dict = OrderedDict()
II_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer 1 to layer 3
II_1_3_parameter_dict = OrderedDict()
II_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer 2 to layer 1
II_2_1_parameter_dict = OrderedDict()
II_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer 2 to layer 3
II_2_3_parameter_dict = OrderedDict()
II_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Excitatory to Excitatory connection from layer 3 to layer x
II_3_1_parameter_dict = OrderedDict()
II_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

II_3_2_parameter_dict = OrderedDict()
II_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(II_delay_line_parameters)
II_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dendritic_arbor_parameters)
II_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(II_dynamical_synapse_parameters)
II_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)


# Excitatory to Inhibitory connections
EI_1_1_parameter_dict = OrderedDict()
EI_1_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_1_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_1_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_1_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_1_2_parameter_dict = OrderedDict()
EI_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_1_3_parameter_dict = OrderedDict()
EI_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

##############
EI_2_1_parameter_dict = OrderedDict()
EI_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_2_2_parameter_dict = OrderedDict()
EI_2_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_2_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_2_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_2_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_2_3_parameter_dict = OrderedDict()
EI_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

#############
EI_3_1_parameter_dict = OrderedDict()
EI_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_3_2_parameter_dict = OrderedDict()
EI_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_3_3_parameter_dict = OrderedDict()
EI_3_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(EI_delay_line_parameters)
EI_3_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dendritic_arbor_parameters)
EI_3_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(EI_dynamical_synapse_parameters)
EI_3_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

# Inhibitory to Excitatory connections
IE_1_1_parameter_dict = OrderedDict()
IE_1_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_1_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_1_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_1_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_1_2_parameter_dict = OrderedDict()
IE_1_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_1_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_1_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_1_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_1_3_parameter_dict = OrderedDict()
IE_1_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_1_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_1_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_1_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

############
IE_2_1_parameter_dict = OrderedDict()
IE_2_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_2_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_2_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_2_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_2_2_parameter_dict = OrderedDict()
IE_2_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_2_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_2_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_2_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_2_3_parameter_dict = OrderedDict()
IE_2_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_2_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_2_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_2_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

##########
IE_3_1_parameter_dict = OrderedDict()
IE_3_1_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_3_1_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_3_1_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_3_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_3_2_parameter_dict = OrderedDict()
IE_3_2_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_3_2_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_3_2_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_3_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

IE_3_3_parameter_dict = OrderedDict()
IE_3_3_parameter_dict["delay_line"] = unique_ID_dict_creator.create_unique_ID_dict(IE_delay_line_parameters)
IE_3_3_parameter_dict["arbor"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dendritic_arbor_parameters)
IE_3_3_parameter_dict["axonal_terminal"] =  unique_ID_dict_creator.create_unique_ID_dict(IE_dynamical_synapse_parameters)
IE_3_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(I_dendritic_Spine_parameters)

# Input dicts
EE_input_1_parameter_dict = OrderedDict()
EE_input_1_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_input_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_input_2_parameter_dict = OrderedDict()
EE_input_2_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_input_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EE_input_3_parameter_dict = OrderedDict()
EE_input_3_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EE_input_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_input_1_parameter_dict = OrderedDict()
EI_input_1_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EI_input_1_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_input_2_parameter_dict = OrderedDict()
EI_input_2_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EI_input_2_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)

EI_input_3_parameter_dict = OrderedDict()
EI_input_3_parameter_dict["axonal_terminal"] = unique_ID_dict_creator.create_unique_ID_dict(EE_dynamical_synapse_parameters)
EI_input_3_parameter_dict["dendritic_spines"] =  unique_ID_dict_creator.create_unique_ID_dict(E_dendritic_Spine_parameters)


if __name__ == '__main__':


    with Client(n_workers = 6) as client:

        '''
        Build neuron
        ######################################################################################
        Here we instantiate each neuron population. The first step of this process is to create
        Neurons objects. Upon initiation these will contain future somas and not much else.
        We can then use the connection dictionaries to create connections between populations

        '''
        #client = Client(asynchronous = True, n_workers = 4, threads_per_worker = 2)

        # Here we first create neuron object. Initioally these only contain the soma of the neurons
        # as they are not connected to anything and thus don't have dendrites or axons
        print("Constructing neurons")
        E1_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, E_1_soma_parameter_dict, 0, "E1_soma", client)
        E2_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, E_2_soma_parameter_dict, 1, "E2_soma", client)
        E3_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, E_3_soma_parameter_dict, 2, "E3_soma", client)

        I1_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, I_1_soma_parameter_dict, 0, "I1_soma", client)
        I2_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, I_2_soma_parameter_dict, 1, "I2_soma", client)
        I3_neurons = rm.Neurons_fully_distributed(rm.Izhikevich_Soma, I_3_soma_parameter_dict, 2, "I3_soma", client)

        print("Constructing inputs")
        E1_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, E_1_input_parameters, 0, "E1_input", client)
        E2_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, E_2_input_parameters, 1, "E2_input", client)
        E3_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, E_3_input_parameters, 2, "E3_input", client)

        I1_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, I_1_input_parameters, 0, "I1_input", client)
        I2_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, I_2_input_parameters, 1, "I2_input", client)
        I3_input = rm.Input_Neurons(rm.Inputs_Distribute_Single_spike, I_3_input_parameters, 2, "I3_input", client)

        # The interface futures funciton is used to connect two neurons and build all the
        # components the connection consists of based on the paramter dict which contains
        # the parameters for each of the components in the connection.
        # THe function only produces futures objects though, which are not themselves connected yet
        print("Interfacing inputs with neurons")
        E1_input.interface_futures(EE_input_1_parameter_dict, E1_neurons)
        #E2_input.interface_futures(EE_input_2_parameter_dict, E2_neurons)
        #E3_input.interface_futures(EE_input_3_parameter_dict, E3_neurons)
        #I1_input.interface_futures(EI_input_1_parameter_dict, I1_neurons)
        #I2_input.interface_futures(EI_input_2_parameter_dict, I2_neurons)
        #I3_input.interface_futures(EI_input_3_parameter_dict, I3_neurons)

        print("Interfacing Neurons with Neurons")
        E1_neurons.interface_futures(self_E_1_parameter_dict, E1_neurons)
        E1_neurons.interface_futures(EE_1_2_parameter_dict, E2_neurons)
        E1_neurons.interface_futures(EE_1_3_parameter_dict, E3_neurons)
        E1_neurons.interface_futures(EI_1_1_parameter_dict, I1_neurons)
        E1_neurons.interface_futures(EI_1_2_parameter_dict, I2_neurons)
        E1_neurons.interface_futures(EI_1_3_parameter_dict, I3_neurons)

        E2_neurons.interface_futures(self_E_2_parameter_dict, E2_neurons)
        E2_neurons.interface_futures(EE_2_1_parameter_dict, E1_neurons)
        E2_neurons.interface_futures(EE_2_3_parameter_dict, E3_neurons)
        E2_neurons.interface_futures(EI_2_1_parameter_dict, I1_neurons)
        E2_neurons.interface_futures(EI_2_2_parameter_dict, I2_neurons)
        E2_neurons.interface_futures(EI_2_3_parameter_dict, I3_neurons)

        E3_neurons.interface_futures(self_E_3_parameter_dict, E3_neurons)
        E3_neurons.interface_futures(EE_3_1_parameter_dict, E1_neurons)
        E3_neurons.interface_futures(EE_3_2_parameter_dict, E2_neurons)
        E3_neurons.interface_futures(EI_3_1_parameter_dict, I1_neurons)
        E3_neurons.interface_futures(EI_3_2_parameter_dict, I2_neurons)
        E3_neurons.interface_futures(EI_3_3_parameter_dict, I3_neurons)

        I1_neurons.interface_futures(self_I_1_parameter_dict, I1_neurons)
        I1_neurons.interface_futures(II_1_2_parameter_dict, I2_neurons)
        I1_neurons.interface_futures(II_1_3_parameter_dict, I3_neurons)
        I1_neurons.interface_futures(IE_1_1_parameter_dict, E1_neurons)
        I1_neurons.interface_futures(IE_1_2_parameter_dict, E2_neurons)
        I1_neurons.interface_futures(IE_1_3_parameter_dict, E3_neurons)

        I2_neurons.interface_futures(self_I_2_parameter_dict, I2_neurons)
        I2_neurons.interface_futures(II_2_1_parameter_dict, I1_neurons)
        I2_neurons.interface_futures(II_2_3_parameter_dict, I3_neurons)
        I2_neurons.interface_futures(IE_2_1_parameter_dict, E1_neurons)
        I2_neurons.interface_futures(IE_2_2_parameter_dict, E2_neurons)
        I2_neurons.interface_futures(IE_2_3_parameter_dict, E3_neurons)

        I3_neurons.interface_futures(self_I_3_parameter_dict, I3_neurons)
        I3_neurons.interface_futures(II_3_1_parameter_dict, I1_neurons)
        I3_neurons.interface_futures(II_3_2_parameter_dict, I2_neurons)
        I3_neurons.interface_futures(IE_3_1_parameter_dict, E1_neurons)
        I3_neurons.interface_futures(IE_3_2_parameter_dict, E2_neurons)
        I3_neurons.interface_futures(IE_3_3_parameter_dict, E3_neurons)

        # Since all components are currently future object we need to call the
        # get component results for the neurons to get the object proxies to be
        # able to call functions on them
        print("Getting Neuron component results")
        E1_neurons.get_component_results()
        E2_neurons.get_component_results()
        E3_neurons.get_component_results()
        I1_neurons.get_component_results()
        I2_neurons.get_component_results()
        I3_neurons.get_component_results()

        print("Getting input component results")
        E1_input.get_component_results()
        E2_input.get_component_results()
        E3_input.get_component_results()
        I1_input.get_component_results()
        I2_input.get_component_results()
        I3_input.get_component_results()

        # the connect components function does what it says and connects the components
        # Internally this is done component by component because the internals of each component
        # is built when they are connected so the results must be gotten for each funciton call
        # before the next can be connected to it. Because of this we don't need to get the results for this call
        print("Connecting neuron components")
        E1_neurons.connect_components()
        E2_neurons.connect_components()
        E3_neurons.connect_components()
        I1_neurons.connect_components()
        I2_neurons.connect_components()
        I3_neurons.connect_components()

        print("Connecting input components")
        E1_input.connect_components()
        #E2_input.connect_components()
        #E3_input.connect_components()
        #I1_input.connect_components()
        #I2_input.connect_components()
        #I3_input.connect_components()

        # Set the number of each cell type in each layer. Here we are setting the number of
        # inhibitory neurons to be 20% by killing 20% of the excitatory neurons in a layer
        # and setting the inhibitory neurons in the same layer to be living at the position
        # of the dead neurons in the excitatory layer.
        percentage_ratio = 0.2
        E1_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        E1_dead_cells[5,10] = 0
        I1_dead_cells = E1_dead_cells == 0
        E2_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        I2_dead_cells = E2_dead_cells == 0
        E3_dead_cells = ncp.random.uniform(0,1,population_size) <percentage_ratio
        I3_dead_cells = E3_dead_cells == 0

        future_E1 = E1_neurons.components[E1_neurons.soma_ID].set_dead_cells(E1_dead_cells)
        future_E2 = E2_neurons.components[E2_neurons.soma_ID].set_dead_cells(E2_dead_cells)
        future_E3 = E3_neurons.components[E3_neurons.soma_ID].set_dead_cells(E3_dead_cells)
        future_I1 = I1_neurons.components[I1_neurons.soma_ID].set_dead_cells(I1_dead_cells)
        future_I2 = I2_neurons.components[I2_neurons.soma_ID].set_dead_cells(I3_dead_cells)
        future_I3 = I3_neurons.components[I3_neurons.soma_ID].set_dead_cells(I3_dead_cells)

        future_E1.result()
        future_E2.result()
        future_E3.result()
        future_I1.result()
        future_I2.result()
        future_I3.result()

        # Create a list of the neuron populatiosn so that we can iterate over them when running the simulation
        neuron_populations = [E1_neurons, E2_neurons, E3_neurons, I1_neurons, I2_neurons, I3_neurons]
        input_populations =[E1_input, E2_input, E3_input, I1_input, I2_input, I3_input]
        '''
        Readout mechanisms
        '''
        '''
        P_delta_readout_parameters = {}
        P_delta_readout_parameters["nr_of_readout_neurons"] = 51
        P_delta_readout_parameters["error_tolerance"] = 0.05
        P_delta_readout_parameters["rho"] = 1 # squashing function boundries
        P_delta_readout_parameters["margin"] = 0.02
        P_delta_readout_parameters["clear_margins_importance"] = 1
        P_delta_readout_parameters["learning_rate"] = 0.0025
        P_delta_readout_parameters["temporal_upper_limit"] = upper_limit
        '''

        '''
        Start simulation
        ######################################################################################
        '''
        spike_trains = ncp.load("spike_trains.npy")
        simulation_length = len(spike_trains[0,:])

        image = ncp.zeros((int(population_size[0]*3), population_size[1], 3))



        nr_of_layers = 6
        network_history = ncp.zeros((population_size[0], population_size[1], nr_of_layers, simulation_length))
        stop_simulation = False

        setling_time = 1 #ms
        setling_time_steps = int(setling_time/time_step)


        for index in range(spike_trains.shape[0]):
            stimulation_train = spike_trains[index,:]
            v = []
            u = []
            spikes_plot = []
            timeline = []
            for t in range(setling_time_steps):
                print("Settling time: ", ncp.round(((t+1)/setling_time_steps)*100,2), "%", end = '\r')
                for population in neuron_populations:
                    population.update_current_values()
                for input in input_populations:
                    input.update_current_values()

                for population in neuron_populations:
                    population.get_results()
                for input in input_populations:
                    input.get_results()

                for population in neuron_populations:
                    population.compute_new_values()
                for input in input_populations:
                    input.compute_new_values(0)


                for population in neuron_populations:
                    population.get_results()
                for input in input_populations:
                    input.get_results()

                image_E1 = E1_neurons.soma.interfacable
                image_E2 = E2_neurons.soma.interfacable
                image_E3 = E3_neurons.soma.interfacable
                image_I1 = I1_neurons.soma.interfacable
                image_I2 = I2_neurons.soma.interfacable
                image_I3 = I3_neurons.soma.interfacable
                network_history[:,:,0,t] = image_E1
                network_history[:,:,1,t] = image_E2
                network_history[:,:,2,t] = image_E3
                network_history[:,:,3,t] = image_I1
                network_history[:,:,4,t] = image_I2
                network_history[:,:,5,t] = image_I3

                E_image = ncp.concatenate((image_E1, image_E2, image_E3), axis = 0)
                I_image = ncp.concatenate((image_I1, image_I2, image_I3), axis = 0)

                image[:,:,2] = E_image
                image[:,:,0] = I_image
                image*=255
                #image = ncp.concatenate((image_E1, image_E2, image_E3, image_I1, image_I2, image_I3), axis = 0)
                #image = inputs.interfacable

                image_shape = ncp.array(image.shape[0:2])
                image_shape *= 15
                image_shape = image_shape[::-1]
                image_shape = tuple(image_shape)

                #image = ncp.asnumpy(image)
                image_out = cv2.resize(image,image_shape)
                image_out = image_out.astype(ncp.uint8)
                cv2.imshow('frame', image_out)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_simulation = True
                    break
            print()
            for t in range(simulation_length):
                # input spikes are generated

                print("t: ", ncp.round(t*time_step,2), " Spike: ", stimulation_train[t], end = '\r')#, ncp.amax(inputs.interfacable))


                for population in neuron_populations:
                    population.update_current_values()
                for input in input_populations:
                    input.update_current_values()

                for population in neuron_populations:
                    population.get_results()
                for input in input_populations:
                    input.get_results()

                for population in neuron_populations:
                    population.compute_new_values()
                for input in input_populations:
                    input.compute_new_values(stimulation_train[t])


                for population in neuron_populations:
                    population.get_results()
                for input in input_populations:
                    input.get_results()



                image_E1 = E1_neurons.soma.interfacable
                image_E2 = E2_neurons.soma.interfacable
                image_E3 = E3_neurons.soma.interfacable
                image_I1 = I1_neurons.soma.interfacable
                image_I2 = I2_neurons.soma.interfacable
                image_I3 = I3_neurons.soma.interfacable
                network_history[:,:,0,t] = image_E1
                network_history[:,:,1,t] = image_E2
                network_history[:,:,2,t] = image_E3
                network_history[:,:,3,t] = image_I1
                network_history[:,:,4,t] = image_I2
                network_history[:,:,5,t] = image_I3

                E_image = ncp.concatenate((image_E1, image_E2, image_E3), axis = 0)
                I_image = ncp.concatenate((image_I1, image_I2, image_I3), axis = 0)

                image[:,:,2] = E_image
                image[:,:,0] = I_image
                image*=255
                #image = ncp.concatenate((image_E1, image_E2, image_E3, image_I1, image_I2, image_I3), axis = 0)
                #image = inputs.interfacable
                #print(ncp.amax(image[:,:,1]))
                image_shape = ncp.array(image.shape[0:2])
                image_shape *= 15
                image_shape = image_shape[::-1]
                image_shape = tuple(image_shape)

                #image = ncp.asnumpy(image)
                image_out = cv2.resize(image,image_shape)
                '''
                voltages = E1_neurons.components["E1_soma"].current_somatic_voltages
                timeline.append((t+1 )*time_step)
                v.append(voltages[5,10])
                u.append(E1_neurons.components["E1_soma"].current_u[5,10])
                spikes_plot.append(image_E1[5,10])
                '''
                #print(ncp.amax(image))
                image_out = image_out.astype(ncp.uint8)
                cv2.imshow('frame', image_out)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    stop_simulation = True
                    break
            for population in neuron_populations:
                population.get_results()


            #cv2.destroyAllWindows()
            '''
            timeline = ncp.array(timeline)
            plt.plot(timeline, ncp.array(v), label = "somatic v")
            plt.plot(timeline, ncp.array(u), label = "recovery u")
            plt.plot(timeline, ncp.array(spikes_plot), label = "spikes")
            plt.legend()
            plt.show()
            #print(v)
            #print(u)
            '''
            if stop_simulation:
                break
        ncp.save("network_history", network_history)
        cv2.destroyAllWindows()
