# print("update")#print("new")# return 2import numpy as np

import cupy as cp

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
from .neural_structure import NeuralStructure

from .somas import BaseIntegrateAndFireSoma, CircuitEquationIntegrateAndFireSoma, IzhikevichSoma

from .dendritic_arbors import DynamicalAxonalTerminalMarkramEtal1998

from .dendritic_spines import DendriticSpineMaas

from .arborizers import DendriticArbor

from .delay_lines import DelayLine

from .neurons import NeuronsFullyDistributed, InputNeurons, NeuronsLocal, InputNeuronsLocal

from .inputs import InputsDistributeSingleSpike

from .readouts import ReadoutPDelta

from .reconstructors import Network

from .help_functions import UniqueIdDictCreator


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

        self.InterfacableArray = InterfacableArray

        self.Component = NeuralStructure

        self.BaseIntegrateAndFireSoma = BaseIntegrateAndFireSoma

        self.CircuitEquationIntegrateAndFireSoma = CircuitEquationIntegrateAndFireSoma

        self.IzhikevichSoma = IzhikevichSoma

        self.DynamicalAxonalTerminalMarkramEtal1998 = DynamicalAxonalTerminalMarkramEtal1998

        self.DendriticSpineMaas = DendriticSpineMaas

        self.DendriticArbor = DendriticArbor

        self.DelayLine = DelayLine

        self.NeuronsFullyDistributed = NeuronsFullyDistributed

        self.InputNeurons = InputNeurons

        self.InputsDistributeSingleSpike = InputsDistributeSingleSpike

        self.ReadoutPDelta = ReadoutPDelta

        self.Network = Network

        self.UniqueIdDictCreator = UniqueIdDictCreator




if __name__ == "__main__":

    print(VERSION)

