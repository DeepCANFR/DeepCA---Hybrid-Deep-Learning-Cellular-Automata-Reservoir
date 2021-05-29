import numpy as ncp

'''
Readouts
'''

class SquashingFunctionRho():
    pass

class ReadoutPDelta(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

        self.nr_of_readout_neurons = self.nr_of_readout_neurons
        self.parallel_perceptron_outputs = ncp.zeros(
            self.nr_of_readout_neurons)

        self.squashing_function = SquashingFunctionRho(self.rho)
        self.margin = self.margin
        # gamma in paper

        self.clear_margin_importance = self.clear_margins_importance
        # mu in paper

        self.error_tolerance = self.error_tolerance
        # small epsilon in paper

        self.learning_rate = self.learning_rate
        # eta in paper

    def activation_function(self):
        input_projection = ncp.repeat(
            self.inputs[:, :, ncp.newaxis], self.nr_of_readout_neurons, axis=2)
        parallel_perceptron_outputs = ncp.sum(
            input_projection*self.weights, axis=(0, 1))
        return parallel_perceptron_outputs

    def update_weights(self, desired_output):
        self.desired_output = desired_output

        # testing fic
        input_projection = ncp.repeat(
            self.inputs[:, :, ncp.newaxis], self.nr_of_readout_neurons, axis=2)
        parallel_perceptron_outputs = ncp.sum(
            input_projection*self.weights, axis=(0, 1))
        #self.parallel_perceptron_outputs *= 0.3
        #self.parallel_perceptron_outputs += parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.parallel_perceptron_outputs

        #parallel_perceptron_outputs = self.activation_function()

        # summary rule 1
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        # adding axis and transposing to allow the array to be multiplied with 3d input array correctly
        parallel_perceptron_output_above_equal_0 = parallel_perceptron_output_above_equal_0[
            ncp.newaxis, ncp.newaxis, :].T.T

        # summary rule 2
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0 = parallel_perceptron_output_below_0[
            ncp.newaxis, ncp.newaxis, :].T.T
        # summary rule 3, note: margin is yotta in paper
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_above_0_below_margin *= parallel_perceptron_output_above_0_below_margin < self.margin
        parallel_perceptron_output_above_0_below_margin = parallel_perceptron_output_above_0_below_margin[
            ncp.newaxis, ncp.newaxis, :].T.T

        # summary rule 4
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_outputs < 0
        parallel_perceptron_output_below_0_above_neg_margin *= parallel_perceptron_outputs > -1*self.margin
        parallel_perceptron_output_below_0_above_neg_margin = parallel_perceptron_output_below_0_above_neg_margin[
            ncp.newaxis, ncp.newaxis, :].T.T

        # summary rule 5
        # zeros
        weight_update_direction = ncp.zeros(self.weight_shape)

        population_output = ncp.sum(
            parallel_perceptron_output_above_equal_0) - ncp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        # compute the lower limits first and then the higher

        if population_output > self.desired_output + self.error_tolerance:

            weight_update_direction += (-1) * input_projection * \
                parallel_perceptron_output_above_equal_0

        elif population_output < self.desired_output - self.error_tolerance:

            masked_input_projection = input_projection * parallel_perceptron_output_below_0
            weight_update_direction += masked_input_projection

        if population_output >= (self.desired_output - self.error_tolerance):
            weight_update_direction += self.clear_margin_importance * \
                (-1 * input_projection) * \
                parallel_perceptron_output_below_0_above_neg_margin

        if population_output <= self.desired_output + self.margin:

            weight_update_direction += self.clear_margin_importance * \
                input_projection * parallel_perceptron_output_above_0_below_margin

        # something strange is happening with the weight update. Testing with random update to see if it is an issue with the accuracy calculation
        #weight_update_direction = ncp.random.uniform(-1,1,self.weight_shape)
        weight_update_direction *= self.learning_rate

        weight_bounding = self.weights.reshape(
            self.weights.shape[0]*self.weights.shape[1], self.weights.shape[2])
        weight_bounding = (ncp.linalg.norm(
            weight_bounding, ord=2, axis=0)**2 - 1)
        # print(weight_bounding)
        weight_bounding = weight_bounding[ncp.newaxis, ncp.newaxis, :].T.T
        weight_bounding *= self.learning_rate
        weight_bounding = self.weights * weight_bounding

        # update weights
        self.weights -= weight_bounding
        self.weights += weight_update_direction

        self.current_population_output = population_output

    def classify(self, image):
        input_projection = ncp.repeat(
            image[:, :, ncp.newaxis], self.nr_of_readout_neurons, axis=2)
        parallel_perceptron_outputs = ncp.sum(
            input_projection*self.weights, axis=(0, 1))

        parallel_perceptron_output_above_equal_0 = parallel_perceptron_outputs >= 0
        parallel_perceptron_output_below_0 = parallel_perceptron_outputs < 0

        population_output = ncp.sum(
            parallel_perceptron_output_above_equal_0) - ncp.sum(parallel_perceptron_output_below_0)
        population_output = self.squashing_function(population_output)

        return population_output

    def interface(self, external_component):
        # read_variable is a 2d array of spikes
        self.external_component = external_component
        external_component_read_variable = self.external_component.interfacable
        external_component_read_variable_shape = external_component_read_variable.shape

        self.inputs = ncp.zeros(external_component_read_variable_shape)

        self.weight_shape = list(self.inputs.shape)
        #print("list weight shape ", self.weight_shape)
        self.weight_shape.append(self.nr_of_readout_neurons)
        #print("appended list weight shape ", self.weight_shape)
        self.weights = ncp.random.uniform(-1, 1, self.weight_shape)

    def update_current_values(self):
        self.inputs = self.external_component.interfacable
