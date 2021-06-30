import numpy as ncp

'''
Support classes
'''


class InterfacableArray(object):
    def __init__(self, population_shape):
        self.array = ncp.zeros(population_shape)
        self.array = self.array[:, :, ncp.newaxis]

        self.external_components = []
        self.external_components_indexes = []

    def interface(self, external_component):

        external_interface = external_component.interfacable
        external_interface_shape = external_interface.shape

        if len(self.external_components) == 0:
            if len(external_interface_shape) == 2:
                self.array = ncp.zeros(external_interface_shape)
                self.array = self.array[:, :, ncp.newaxis]
                self.external_components_indexes.append(slice(0, 1, 1))

            else:
                self.array = ncp.zeros(external_interface_shape)
                self.external_components_indexes.append(
                    slice(0, external_interface_shape[2], 1))

        else:
            if len(external_interface_shape) == 2:
                old_axis_2_length = self.array.shape[2]
                self.array = ncp.concatenate(
                    (self.array, external_interface[:, :, ncp.newaxis]), axis=2)
                self.external_components_indexes.append(
                    slice(old_axis_2_length, self.array.shape[2], 1))

            else:
                old_axis_2_length = self.array.shape[2]
                self.array = ncp.concatenate(
                    (self.array, external_interface), axis=2)
                self.external_components_indexes.append(
                    slice(old_axis_2_length, self.array.shape[2], 1))

        self.external_components.append(external_component)

    async def update(self):

        for index, external_component in enumerate(self.external_components):
            component_index = self.external_components_indexes[index]
            external_interface = external_component.interfacable

            if len(external_interface.shape) == 2:
                external_interface = external_interface[:, :, ncp.newaxis]
                self.array[:, :, component_index] = external_interface
            else:
                self.array[:, :, component_index] = external_interface

        #print(self.array.shape, ncp.amax(self.array))
    def get_sum(self):
        return ncp.sum(self.array, axis=2)
