# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 18:42:04 2021

@author: trymlind
"""


import cupy as cp
import numpy as np
import time
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import shared_memory
import sys
import dask as dask



if False:
    sim_time = 1000
    population_size = (1000,1000)
    delta_t = 1
    time_since_last_spike = cp.zeros(population_size)

    spikes = cp.random.rand(population_size[0], population_size[1], sim_time) < 0.1

    start_time = time.time()
    for t in range(sim_time):
        time_since_last_spike += delta_t
        time_since_last_spike *= spikes[:,:,t] == 0
    print("*= inverse spikes: ", time.time() - start_time)

    time_since_last_spike = cp.zeros(population_size)
    start_time = time.time()
    for t in range(sim_time):
        time_since_last_spike += delta_t
        spike_indexes = cp.where(spikes[:,:,t] == 0)
        time_since_last_spike[spike_indexes[0], spike_indexes[1]] = 0
    print("cp.where: ", time.time() - start_time)


if False:
    x = cp.array([1000000.0])
    while x <= 1000000:
        x *= 0.1
        print(x)

if False:
    class Exponential_decay(V,t):
        def __init__(self):
            self.last_max_V = 0
            self.current_V
            self.new_V
            self.t_since_spike = 0
            self.time_step
        def compute_new_value(inputs):
            self.last_max_V += inputs
            if inputs > 0:
                self.time_since_last_spike = 0
            else:
                self.time_since_last_spike += self.time_step

if False:
    resting_utilization_of_synaptic_efficacy = -2
    current_utilization_of_synaptic_efficacy = 1.2
    new_utilization_of_synaptic_efficacy = 1.3
    current_neurotransmitter_reserve = 1.2
    tau_facil = 30
    tau_recovery = 30

    time_since_last_spike = 0

    sim_t = 100
    r = np.zeros(sim_t)
    u = np.zeros(sim_t)
    t = np.zeros(sim_t)
    for i in range(sim_t):

        new_utilization_of_synaptic_efficacy = current_utilization_of_synaptic_efficacy * cp.exp((-time_since_last_spike) / tau_facil) + resting_utilization_of_synaptic_efficacy*(1 - current_utilization_of_synaptic_efficacy * cp.exp((-time_since_last_spike) / tau_facil))
        new_neurotransmitter_reserve = current_neurotransmitter_reserve * (1 - new_utilization_of_synaptic_efficacy)*cp.exp(-time_since_last_spike / tau_recovery) + 1 - cp.exp(-time_since_last_spike / tau_recovery)

        current_utilization_of_synaptic_efficacy = new_utilization_of_synaptic_efficacy
        current_neurotransmitter_reserve = new_neurotransmitter_reserve
        time_since_last_spike += 1
        r[i] = current_neurotransmitter_reserve
        u[i] = current_utilization_of_synaptic_efficacy
        t[i] = cp.exp((-time_since_last_spike) / tau_facil)
    plt.figure(1)
    plt.plot(r)
    plt.plot(u)
    plt.plot(t)

if False:

    x = cp.array([0,1,2,3])
    y = cp.array([0,0,0,0.])

    class Do(mp.Process):
        def __init__(self, read_variable_name, output_variable):


            self.read_variable = cp.ndarray(read_variable_name.shape, dtype = read_variable_name.dtype, memptr = read_variable_name.data)
            self.output_variable = cp.ndarray(output_variable.shape, dtype = output_variable.dtype, memptr = output_variable.data)



        def do_something(self):
            print("doing", self.read_variable, " + ", self.output_variable)
            self.output_variable += self.read_variable
            print("results: ", self.output_variable)
            return self.output_variable
            sys.stdout.flush()

    do_1 = Do(x,y)
    do_2 = Do(x,y)

    if __name__ == "__main__":
        p1 = mp.Process(target = do_1.do_something)
        p2 = mp.Process(target = do_2.do_something)

        p1.start()
        p2.start()

        p1.join()
        p2.join()

    print(do_1.output_variable)
    print(do_2.output_variable)
    print(y)
    '''
    if __name__ == '__main__x':
        print("parall")
        network_compute_new_processes = []
        network_update_current_values_processes = []

        for component in network:
            network_compute_new_processes.append(multiprocessing.Process(target = component.compute_new_values))
            network_update_current_values_processes.append(multiprocessing.Process(target = component.update_current_values))

        for component in network_compute_new_processes:
            component.start()
        for component in network_compute_new_processes:
            component.join()
    else:
        print("failed to run parallel code")
        sys.exit(0)
    '''

if False:
    class Component_process(mp.Process):
        def __init__(self, control_connection):
            super(Component_process,self).__init__()
            self.control_connection = control_connection

        def run(self):
            order = self.control_connection.recv()
            process_running = True
            print("order received")
            #print(start_order)
            #sys.stdout.flush()
            while process_running:

                if order == "compute_new":
                    self.read_variable = self.read_connection.recv()
                    self.read_variable += 1


                    print("computing_new ", 1)
                    self.control_connection.send("computed_new")
                elif order == "update_old":
                    self.write_pipe_parent.send(self.read_variable)
                    self.control_connection.send("updated_old")
                elif order == "stop":
                    print(self.read_variable)
                    print("I'm done")
                    break
                order = self.control_connection.recv()


        def interface_read_variable(self, read_variable_pipe):
            self.read_connection = read_variable_pipe
            self.read_variable = self.read_connection.recv()
            self.output_variable = np.zeros(self.read_variable.shape)


        def interface_write_variable(self):
            self.write_pipe_parent, write_pipe_child = mp.Pipe()
            return write_pipe_child


    if __name__ == "__main__":
        parent_conn_1, child_conn_1 = mp.Pipe()
        parent_conn_2, child_conn_2 = mp.Pipe()

        p1 = Component_process(child_conn_1)
        p2 = Component_process(child_conn_2)

        inputs = np.array([1])

        p1.interface_read_variable()
        p1.start()
        p2.start()
        print("processes started")
        for i in range(10):
            print("sending compute new")
            parent_conn_1.send("compute_new")
            parent_conn_2.send("compute_new")
            print("sendt")
            if parent_conn_1.recv() == "computed_new":
                print("sending update 1")
                parent_conn_1.send("update_old")
                print("sendt")
            if parent_conn_2.recv() == "computed_new":
                print("sending update 2")
                parent_conn_2.send("update_old")
                print("sendt")
        parent_conn_1.send("stop")
        parent_conn_2.send("stop")




        #p1.join()
        #p2.join()




if False:
    conn_1, conn_2 = mp.Pipe()

    x = np.zeros(1)
    conn_1.send(x)
    y = conn_2.recv()
    print("success with 1")

    # this is too big
    x = np.zeros((1000,1000))
    conn_1.send(x)
    y = conn_2.recv()
    print("success with large array")


if False:




    class Component(mp.Process):
        def __init__(self,names):
            super(Component,self).__init__()
            self.names = names
        def interface_read_variable(self, x, shm_name):
            shm = mp.shared_memory.SharedMemory(name = shm_name)
            self.interfaced_variable = np.ndarray(x.shape, dtype = x.dtype, buffer = shm.buf)
            self.interfaced_variable[:,:] = x[:,:]

            array = np.zeros(x.shape)
            self.output_mem = mp.shared_memory.SharedMemory(create = True, size = array.nbytes)
            self.output_variable = np.ndarray(array.shape, array.dtype, buffer = self.output_mem.buf)
            self.output_variable[:,:] = array[:,:]


        def get_output_mem(self):
            return self.output_mem.name

        def run(self):

            print("name: ",self.names, "\n","pre output \n",self.output_variable, "\n", "pre read \n", self.interfaced_variable, "\n\n")

            time.sleep(1)


            self.output_variable[1,int(self.interfaced_variable[1,1])] = 1 + self.interfaced_variable[1,int(self.interfaced_variable[1,1])]

            print("name: ",self.names, "\n","post output \n", self.output_variable, "\n", "post read \n", self.interfaced_variable, "\n\n")


    if __name__ == "__main__":
        x = np.ones((5,5))
        shm = mp.shared_memory.SharedMemory(create = True, size = x.nbytes)
        b = np.ndarray(x.shape, dtype = x.dtype, buffer = shm.buf)
        b[:] = x[:]

        component_1 = Component("rt1")
        component_2 = Component("rt2")

        component_1.interface_read_variable(b,shm.name)
        print("b \n", b)
        b[:] = x[:]
        component_2.interface_read_variable(component_1.output_variable, component_1.get_output_mem())
        print("b \n", b)

        shm_1 = mp.shared_memory.SharedMemory(name = component_1.output_mem.name)
        shm_2 = mp.shared_memory.SharedMemory(name = component_2.output_mem.name)
        out_1 = np.ndarray(shape = component_1.output_variable.shape, dtype = component_1.output_variable.dtype, buffer = shm_1.buf)
        out_2 = np.ndarray(shape = component_2.output_variable.shape, dtype = component_2.output_variable.dtype, buffer = shm_2.buf)

        component_1.start()
        time.sleep(1)
        component_2.start()

        time.sleep(3)
        print(out_1)
        print(out_2)

        time.sleep(3)

        print("b \n", b)

'''
Testing dask for parallelization
'''

if False:


    def inc(x):
        time.sleep(1)
        x += 1
        return x

    def sum(x,y):
        time.sleep(1)
        z = x + y
        return z

    x = dask.delayed(inc)(1)
    y = dask.delayed(inc)(3)
    z = dask.delayed(sum)(x,y)

    #z.visualize()
    z_results = z.compute()
    print(z_results)

    ####
if False:
    # objects doesnt' work for some reason. The values of the array contained in the object has all wrong values
    x = np.arange(3)
    x = np.repeat(x[:,np.newaxis],3,axis = 1)
    print(x)
    shifts = [(0,0),(1,1),(0,1),(1,0)]

    class Input_Array(object):

        def __init__(self, shape, source_array, shifts):
            self.projection_3d = np.zeros(shape)
            self.source_array = source_array
            self.shifts = shifts

        def project_2d_array(self):

            for index, shift in enumerate(self.shifts):
                self.projection_3d[:,:,index] = self.source_array

        @property
        def get_projection_3d(self):
            #self.project_2d_array()
            return self.projection_3d

    input_array_1 = dask.delayed(Input_Array((3,3,4),x,shifts))
    input_array_2 = dask.delayed(Input_Array((3,3,4),x,shifts))
    i_list = [input_array_1, input_array_2]

    '''
    for thing in i_list:
        thing.project_2d_array()
    print(np.sum(input_array_1.projection_3d, axis = 2) + np.sum(input_array_2.projection_3d, axis = 2))

    input_array_1 = Input_Array((3,3,4))
    input_array_2 = Input_Array((3,3,4))
    i_list = [input_array_1, input_array_2]
    '''

    ia_1 = dask.delayed(input_array_1.project_2d_array)
    ia_2 = dask.delayed(input_array_2.project_2d_array)

    results = dask.delayed(ia_1)() + dask.delayed(ia_2)()
    output = results.compute()
    get_1 = dask.delayed(lambda ia_1: input_array_1.get_projection_3d, pure = False)
    get_2 = dask.delayed(lambda ia_2: input_array_2.get_projection_3d, pure = False)

    #soma_input = dask.delayed(np.sum)(get_1(input_array_1) ,axis = 2) + dask.delayed(np.sum)(get_2(input_array_2) ,axis = 2)
    #print(soma_input.compute())
    print(get_1(input_array_1).compute().compute())

if False:
    def put_shift(array_3d, array_2d, shifts):
        for index, shift in enumerate(shifts):
            array_3d[:,:,index] = np.roll(array_2d, shift)
        return array_3d

    array_3d_1 = np.empty((3,3,4))
    array_3d_2 = np.empty((3,3,4))
    arrays_3d = [array_3d_1, array_3d_2]

    x = np.arange(3)
    x = np.repeat(x[:,np.newaxis],3,axis = 1)

    shifts = [(0,0),(1,1),(0,1),(1,0)]
    finished = []
    for array in arrays_3d:
        out = dask.delayed(put_shift)(array,x,shifts)
        finished.append(out)

    sum = np.sum(finished[0]) + np.sum(finished[1])
    print(sum.compute())

if False:

    class Incrementer(object):
        def __init__(self):
            self._n = 0
            self.a = np.zeros(10)
        @property
        def n(self):
            self._n += 1
            x = self._n
            self.a[x] = self._n
            return self.a

    x = dask.delayed(Incrementer())
    y = dask.delayed(Incrementer())
    x.n.key == x.n.key
    True
    get_n = dask.delayed(lambda x: x.n, pure=False)
    #get_n = dask.delayed(lambda x: y.n, pure=False)

    #results = get_n_x + get_n_y
    #print(results.compute())
    print(get_n(x).compute())
    print(get_n(x).compute())
    print(get_n(x).compute())

if True:
    class Chain_link(object):
        def __init__(self):
            self.receiver = 0
            self.sender = 0
            self.delayed_input = False

        @dask.delayed
        def update_sender(self):
            x = self.receiver
            self.sender = x
            time.sleep(2)
            return 1
        @dask.delayed
        def update_receiver(self):
            time.sleep(2)
            if self.delayed_input:
                self.receiver = self.get_input(self.input_object).compute()
            else:
                self.receiver = self.input_object#.compute()
            return 1

        def interface(self, input_object, delayed_input):

            self.delayed_input = delayed_input
            if delayed_input:
                self.input_object = input_object
                self.get_input = dask.delayed(lambda input_object: self.input_object.sender, pure = False)
            else:
                self.input_object = input_object

    def Inputs(object):
        def __init__(self, starting_input):
            self.input =

    nr_of_links = 6
    chain = []
    for i in range(nr_of_links):
        link = Chain_link()
        chain.append(link)

    inputs = 1
    chain[0].interface(inputs, False)
    for i in range(1,nr_of_links):
        chain[i].interface(chain[i-1], True)

    for i in range(nr_of_links):
        if i == 2:
            inputs -= 1
        t = time.time()
        print("\n new time step \n")
        total_completed = []
        for i in range(nr_of_links):
            finished = chain[i].update_receiver()
            total_completed.append(finished)
        sum_completed = dask.delayed(sum)(total_completed)
        print(sum_completed.compute())
        for link in chain:
            print(link.receiver, link.sender)

        total_completed = []
        for i in range(nr_of_links):
            finished = chain[i].update_sender()
            total_completed.append(finished)
        sum_completed = dask.delayed(sum)(total_completed)
        print(sum_completed.compute())
        for link in chain:
            print(link.receiver, link.sender)
        print(time.time() - t)

if False:
    # it does not work to not use a delayed function to get the updated self.sender values
    class Chain_link(object):
        def __init__(self):
            self.receiver = 0
            self.sender = 0
            self.delayed_input = False

        @dask.delayed
        def update_sender(self):
            x = self.receiver
            self.sender = x
            time.sleep(2)
            return 1
        @dask.delayed
        def update_receiver(self):
            time.sleep(2)
            self.receiver = self.current_input
            return 1

        def interface(self, inputs):

            self.current_input = inputs
    nr_of_links = 6
    chain = []
    for i in range(nr_of_links):
        link = Chain_link()
        chain.append(link)

    inputs = 1
    chain[0].interface(inputs)
    for i in range(1,nr_of_links):
        chain[i].interface(chain[i-1].sender)

    for i in range(nr_of_links):
        if i == 2:
            inputs -= 1
        t = time.time()
        print("\n new time step \n")
        total_completed = []
        for i in range(nr_of_links):
            finished = chain[i].update_receiver()
            total_completed.append(finished)
        sum_completed = dask.delayed(sum)(total_completed)
        print(sum_completed.compute())
        for link in chain:
            print(link.receiver, link.sender)

        total_completed = []
        for i in range(nr_of_links):
            finished = chain[i].update_sender()
            total_completed.append(finished)
        sum_completed = dask.delayed(sum)(total_completed)
        print(sum_completed.compute())
        for link in chain:
            print(link.receiver, link.sender)
        print(time.time() - t)
