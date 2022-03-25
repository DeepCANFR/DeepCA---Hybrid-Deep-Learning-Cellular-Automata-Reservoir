from .nodes import Node 
import numpy as np

class ConwaysGameOfLife(Node):
    def __init__(self, parameters):
        super().__init__(parameters)
        world_size = parameters["world_size"]
        on_locations_starting_state = parameters["on_locations_starting_state"]

        self.current_state.update({
            "world":np.zeros(world_size)
        })
        self.copy_next_state_from_current_state()
        

        if len(on_locations_starting_state) > 0:
            on_locations_starting_state = np.array(on_locations_starting_state)
            self.current_state["world"][on_locations_starting_state] = 1

        self.connections = [(1,1),(0,1),(-1,1),
                            (1,0),(-1,0),
                            (1,-1),(0,-1),(-1,-1)]

    def compute_next(self):
        current_world = self.current_state["world"]
        next_world = self.next_state["world"]
        world_size = self.parameters["world_size"]

        connection_array = np.zeros((world_size[0], world_size[1], len(self.connections)))

        for index, connection in enumerate(self.connections):
            connection_array[:,:, index] = np.roll(current_world, connection, axis = (0,1))
        
        summed_connections = np.sum(connection_array,2)

        # any dead cell with three live neighbours comes alive
        dead_cells = current_world == 0

        cells_with_two_or_more_live_neighbours = summed_connections >= 2
        cells_with_fewer_than_4_live_neighbours = summed_connections < 4
        cells_with_3_live_neighbours = summed_connections == 3

        live_cells_with_two_or_three_neighbours = current_world*cells_with_two_or_more_live_neighbours*cells_with_fewer_than_4_live_neighbours
        dead_cells_with_3_live_neighbours = dead_cells*cells_with_3_live_neighbours

        next_world[:,:] = live_cells_with_two_or_three_neighbours + dead_cells_with_3_live_neighbours
        






