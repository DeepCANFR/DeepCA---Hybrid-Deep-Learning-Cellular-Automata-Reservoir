'''

    Delay lines

'''


import numpy as np
import sys
from .nodes import Node

class DelayLineNode(Node):
    '''
    DelayLines simulates simplified axons. Their input is an array of spikes from some source. 
    The delay is implemented as a population_size + delay_in_time_steps shaped array. 
    '''
    def __init__(self, parameters):
        super().__init__(parameters)
        population_size = self.parameters["population_size"]
        
        # current state with next
        self.current_state["spike_output"] = np.zeros(population_size)
        self.copy_next_state_from_current_state()

        # current without next
        self.current_state.update({
            "spike_source":np.zeros(population_size),
            "time_step_nr":np.array(0),

        })

        # find length of delay in timesteps
        time_step = self.parameters["time_step"]
        delay = self.parameters["delay"]
        delay_in_time_steps = delay/time_step

        # for all inputs to be stored in a delay line the number of indexes must be equal to the number of 
        # time_steps of the delay. If this is not an integer number we cannot create an accurate delay
        # because we cannot have a non-integer number of indexes over the time axis
        # therefor the user is prompted to choose different values
        if not delay_in_time_steps.is_integer():
            print("delay / time_step produced non integer number. This will result in inaccurate delay time. Please choose different time-step or delay")
            sys.exit(0)
        else:
            delay_line_shape = [delay_in_time_steps]
            print(delay_line_shape)
            population_size_list = list(population_size)
            delay_line_shape = delay_line_shape + population_size_list
            print(delay_line_shape)
            delay_line_shape = tuple([int(i) for i in delay_line_shape])
            print(type(delay_line_shape))
            self.current_state.update({
                                "delay_line":np.zeros(delay_line_shape),
                                
            })

        # static state
        self.static_state.update({
            "delay_in_time_steps":np.array(delay_in_time_steps)
        })

    def compute_next(self):
        spike_source = self.current_state["spike_source"]
        delay_line = self.current_state["delay_line"]
        time_step_nr = self.current_state["time_step_nr"] # maybe not use array? How long will people simulate for?
        
        spike_output = self.next_state["spike_output"]
        
        delay_in_time_steps = self.static_state["delay_in_time_steps"]

        # ToDo: create implementation that doesn't use roll, but indexing instead (increasing indexes, see tryouts.py)
        next_delay_line = np.roll(delay_line, 1, axis = 0)
        next_delay_line[0,:,:] = spike_source[:,:] # ToDo: make it so it works with arbitrary shape (keep time to axis 0)

        np.copyto(delay_line, next_delay_line)

        np.copyto(spike_output, delay_line[-1,:,:])

