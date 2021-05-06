#print("update")#print("new")# return 2import numpy as np
import cupy as cp
import numpy as ncp
# Note: ncp is supposed to offer the option of using either numpy or cupy as dropin for computations
import dask
import string
import random
import time
import sys
from collections import OrderedDict
from differential_equation_solvers import RungeKutta2_cupy, ForwardEuler_cupy
from membrane_equations import IntegrateAndFireNeuronMembraneFunction, CircuitEquation, IzhivechikEquation
from spike_generators import PoissonSpikeGenerator
from support_classes import Interfacable_Array
from component import Component 
from somas import BaseIntegrateAndFireSoma, CircuitEquationIntegrateAndFireSoma, IzhikevichSoma
from dendritic_arbors import DynamicalAxonalTerminalMarkramEtal1998
from dendritic_spines import DendriticSpineMaas
from arborizers import DendriticArbor
from delay_lines import DelayLine
from neurons import NeuronsFullyDistributed, InputNeurons
from inputs import InputsDistributeSingleSpike
from readouts import ReadoutPDelta 
from reconstructors import Network
from help_functions import UniqueIdDictCreator

VERSION = "0.0.1"



if __name__ == "__main__":
    print(VERSION)