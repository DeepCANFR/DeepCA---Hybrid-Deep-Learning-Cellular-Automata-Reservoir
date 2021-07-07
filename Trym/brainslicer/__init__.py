# print("update")#print("new")# return 2import numpy as np

#import cupy as cp

import numpy as np

# Note: ncp is supposed to offer the option of using either numpy or cupy as dropin for computations

import dask

import string
import random
import time

import sys

from collections import OrderedDict

#import .differential_equation_solvers
from .differential_equation_solvers import RungeKutta2, ForwardEuler

from .membrane_equations import IntegrateAndFireNeuronMembraneFunction, CircuitEquation, IzhivechikEquation

from .spike_generators import PoissonSpikeGenerator

from .support_classes import InterfacableArray

from .neural_structure import  NeuralStructureNode

from .somas import IzhikevichNode, CircuitEquationNode

from .dendritic_arbors import DynamicalAxonalTerminalMarkramEtal1998Node

from .dendritic_spines import DendriticSpineMaasNode

from .arborizers import  ArborizerNode

from .delay_lines import  DelayLineNode

from .readouts import ReadoutPDelta

from .help_functions import UniqueIdDictCreator

from .graphs import DistributedGraph

from .graph_functions import copy_graph_genome

from .inputs import StaticInput


VERSION = "0.0.1"


class Brainslicer():

    def __init__(self, array_provider=np):

        self.array_provider = array_provider

        self.RungeKutta2_cupy = RungeKutta2

        self.ForwardEuler_cupy = ForwardEuler

        self.IntegrateAndFireNeuronMembraneFunction = IntegrateAndFireNeuronMembraneFunction

        self.CircuitEquation = CircuitEquation

        self.IzhivechikEquation = IzhivechikEquation

        self.PoissonSpikeGenerator = PoissonSpikeGenerator

        self.NeuralStructureNode = NeuralStructureNode

        self.DynamicalAxonalTerminalMarkramEtal1998Node = DynamicalAxonalTerminalMarkramEtal1998Node

        self.DendriticSpineMaasNode = DendriticSpineMaasNode

        self.ArborizerNode = ArborizerNode
        
        self.DelayLineNode = DelayLineNode

        self.ReadoutPDelta = ReadoutPDelta

        self.UniqueIdDictCreator = UniqueIdDictCreator

        self.DistributedGraph = DistributedGraph

        self.IzhikevichNode = IzhikevichNode

        self.CircuitEquationNode = CircuitEquationNode

        self.copy_graph_genome = copy_graph_genome

        self.StaticInput = StaticInput




if __name__ == "__main__":
    # todo: add list of names not to be used when creating new classes and genomes
    print(VERSION)

