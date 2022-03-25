import numpy as ncp
'''
Membrane equations
'''


class IntegrateAndFireNeuronMembraneFunction(object):
    def __init__(self, leakage_reversal_potential, membrane_resistance, membrane_time_constant, summed_inputs):
        self.leakage_reversal_potential = leakage_reversal_potential    # E_m
        self.membrane_resistance = membrane_resistance                  # R_m
        self.membrane_time_constant = membrane_time_constant            # tau_m
        self.summed_inputs = summed_inputs

    def __call__(self, V, t):

        delta_V = (self.leakage_reversal_potential - V + self.membrane_resistance *
                   self.summed_inputs)/self.membrane_time_constant

        return delta_V


class CircuitEquation(object):
    def __init__(self, static_state, current_state):
        self.static_state = static_state 
        self.current_state = current_state


    def __call__(self, V, t):
        input_resistance = self.static_state["input_resistance"]
        summed_inputs = self.current_state["summed_inputs"]
        background_current = self.static_state["background_current"]
        time_constant = self.static_state["time_constant"]

        delta_V = (input_resistance * (summed_inputs +
                   background_current) - V) / time_constant

        return delta_V


class IzhivechikEquation(object):
    def __init__(self, static_state, current_state, parameters):
        self.static_state = static_state
        self.current_state = current_state 
        self.parameters = parameters

    def __call__(self, v_u, t=0):
        a = self.static_state["membrane_recovery"]
        b = self.static_state["resting_potential"]

        summed_inputs = self.current_state["summed_inputs"]
        population_size = self.parameters["population_size"]


        delta_v_u = ncp.zeros(
            (population_size[0], population_size[1], 2))
        delta_v_u[:, :, 0] = 0.04*v_u[:, :, 0]**2 + 5 * \
            v_u[:, :, 0] + 140 - v_u[:, :, 1] + summed_inputs
        delta_v_u[:, :, 1] = a * (b * v_u[:, :, 0] - v_u[:, :, 1])
        return delta_v_u
