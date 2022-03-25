import numpy as ncp
import numpy as np
import time

from .nodes import Node
import sys

class NeuralStructureNode(Node):
    def __init__(self, parameters):
        super().__init__(parameters)

       
    def create_distribution_values(self, distribution, population_size):
        distribution_type = distribution["distribution_type"]
        parameters = distribution["distribution_parameters"]
        if distribution_type == "homogenous":
            arr = np.ones(population_size) * parameters["value"]
            return arr
        elif distribution_type == "normal":
            arr = np.random.normal(**parameters, size = population_size)
            return arr
        elif distribution_type == "uniform":
            arr = np.random.uniform(**parameters, size = population_size)
            return arr
        elif distribution_type == "Izhikevich":
            arr = np.zeros(population_size)
            arr += parameters["base_value"]

            random_variable = np.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            multiplier = parameters["multiplier_value"]
            variance = multiplier * random_variable 

            arr += variance 
            return arr
    
    def remove_negative_or_positive_values(self, arr, distribution, values_to_remove):
        distribution_type = distribution["distribution_type"]
        parameters = distribution["distribution_parameters"]

        population_size = arr.shape

        if values_to_remove == "negative":
            values = arr <= 0 
        elif values_to_remove == "positive":
            values = arr >= 0
        
        if distribution_type == "homogenous":
            print("distribution_type was homogenous, there should be no reason to remove values from this array. Check parameters")
            sys.exit(0)
        elif distribution_type == "normal":
            loc = parameters["loc"]
            scale = parameters["scale"]

            lower_limit = loc - scale
            upper_limit = loc + scale

            if values_to_remove == "positive":
                if upper_limit >= 0:
                    upper_limit = np.nextafter(0.0,-1)

            elif values_to_remove == "negative":
                if lower_limit <= 0:
                    lower_limit = np.nextafter(0.0,1)
            
            replacement_values = np.random.uniform(lower_limit, upper_limit, population_size)
            
            arr *= values == 0
            arr += values * replacement_values
            

    def create_dependent_distribution_values(self, independent_distribution, dependent_distribution, population_size):
        if independent_distribution["distribution_type"] == "Izhikevich" and dependent_distribution["distribution_type"] == "Izhikevich":
            random_variable = np.random.uniform(0,1,population_size)
            random_variable = random_variable**2
            

            independent_parameters = independent_distribution["distribution_parameters"]

            independent_arr = np.zeros(population_size)
            independent_arr += independent_parameters["base_value"]

            independent_multiplier = independent_parameters["multiplier_value"]
            independent_variance = independent_multiplier * random_variable 

            independent_arr += independent_variance 


            dependent_parameters = dependent_distribution["distribution_parameters"]
            dependent_arr = np.zeros(population_size)
            dependent_arr += dependent_parameters["base_value"]

            dependent_multiplier = dependent_parameters["multiplier_value"]
            dependent_variance = random_variable * dependent_multiplier

            dependent_arr += dependent_variance

            return independent_arr, dependent_arr

    def cap_array(self, array, upper_limit):
        below_upper_limit = array < upper_limit
        array *= below_upper_limit
        array += (below_upper_limit == 0)*upper_limit
        return array

