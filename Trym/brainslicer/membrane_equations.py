
'''
Membrane equations
'''
class IntegrateAndFireNeuronMembraneFunction(object):
    def __init__(self, leakage_reversal_potential, membrane_resistance, membrane_time_constant, summed_inputs):
        self.leakage_reversal_potential = leakage_reversal_potential    # E_m
        self.membrane_resistance = membrane_resistance                  # R_m
        self.membrane_time_constant = membrane_time_constant            # tau_m
        self.summed_inputs = summed_inputs

    def __call__(self,V,t):

        delta_V = (self.leakage_reversal_potential - V + self.membrane_resistance * self.summed_inputs)/self.membrane_time_constant

        return delta_V

class CircuitEquation(object):
    def __init__(self, input_resistance, time_constant, summed_inputs, constant_input = 0):

        self.input_resistance = input_resistance
        self.time_constant = time_constant
        self.summed_inputs = summed_inputs

        self.constant_input = constant_input

    def __call__(self, V,t):

        delta_V = (self.input_resistance * (self.summed_inputs + self.constant_input) - V) / self.time_constant

        return delta_V

class IzhivechikEquation(object):
    def __init__(self, a, b, summed_inputs, population_size):
        self.a = a
        self.b = b

        self.summed_inputs = summed_inputs
        self.population_size = population_size

    def __call__(self, v_u, t = 0):

        delta_v_u = ncp.zeros((self.population_size[0], self.population_size[1], 2))
        delta_v_u[:,:,0] = 0.04*v_u[:,:,0]**2 + 5*v_u[:,:,0] + 140 - v_u[:,:,1] + self.summed_inputs
        delta_v_u[:,:,1] = self.a * (self.b * v_u[:,:,0] - v_u[:,:,1])
        return delta_v_u
