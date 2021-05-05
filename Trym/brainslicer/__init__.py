#print("update")#print("new")# return 2import numpy as np
import cupy as cp
import numpy as ncp
# Note: ncp is supposed to offer the option of using either numpy or cupy as dropin for computations
import dask
import string
import random
import time
from collections import OrderedDict
import sys
from differential_equation_solvers import RungeKutta2_cupy, ForwardEuler_cupy
from membrane_equations import Integrate_and_fire_neuron_membrane_function, Circuit_Equation, Izhivechik_Equation
from spike_generators import Poisson_Spike_Generator
from support_classes import Interfacable_Array
from component import Component, 
from somas import Base_Integrate_and_Fire_Soma, Circuit_Equation_Integrate_and_Fire_Soma, Izhikevich_Soma
from dendritic_arbors import Dynamical_Axonal_Terminal_Markram_etal_1998
from dendritic_spines import Dendritic_Spine_Maas
from arborizers import Dendritic_Arbor
from delay_lines import Delay_Line
from neurons import Neurons_fully_distributed, Input_Neurons
from inputs import Inputs_Distribute_Single_spike
from readouts import Readout_P_Delta 
from reconstructors import Network 
from help_functions import Unique_ID_Dict_Creator

VERSION = "0.0.1"



if __name__ == "__main__":
    print(VERSION)