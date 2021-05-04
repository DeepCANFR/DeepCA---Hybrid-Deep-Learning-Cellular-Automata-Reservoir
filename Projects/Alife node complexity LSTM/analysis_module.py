
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

def gaussian_convolution(pattern, tau = 5, time_averaged = True):
    pattern = pattern.astype(np.float64)

    # tau = 5 ms

    # if I have understood these properply the t array gives you the time points
    # the gaussian is placed. So in our case a value of 1 = a spike, is centered at t = 0
    # the gaussian then distributes itself from -25 to 25 time points with the peak centered at the spike
    t_start = -25
    t_end = 25
    t_total = t_end - t_start
    t = np.linspace(t_start,t_end,t_total)
    gaussian = np.exp(-(t/tau)**2)

    # convolve input pattern with the gaussian
    convolved_pattern = np.convolve(pattern, gaussian, mode = "full")
    return convolved_pattern[np.abs(t_start+1):convolved_pattern.shape[0]-t_end]


def d(u,v, sampling_frequency, time_averaged = True):
    # the time time dimension must be on the 0th axis
    if u.shape != v.shape:
        return "Error: input patterns of different shapes"
    else:

        norm = np.linalg.norm(u-v, ord = 2)
        if time_averaged:
            # since the two input patterns are of the same shape and the time axis
            # is on the 0th axis we can divide the number of points along the 0th axis
            # with the sampling frequnecy to get the time length of the input
            # We then divide the norm by this number
            return norm/(u.shape[0]/sampling_frequency)
        else:
            return norm
        # to do: Add spike density averaging


def flatten_high_dimensional_state_history(state_history, time_axis):
    state_history_shape = state_history.shape
    time_length = state_history_shape[time_axis]
    state_history_shape =  np.array(state_history_shape)
    state_history_shape[time_axis] = 1
    state_axis_length =  np.product(state_history_shape)
    output_array = np.zeros((time_length, state_axis_length))

    index = []
    for i in range(len(state_history_shape)):
        index.append(slice(0,None))
    for t in range(time_length):
        index[time_axis] = t
        t_index = tuple(index)
        output_array[t,:] = state_history[t_index].flatten()
    return output_array

def historywise_separability(liquid_1_history, liquid_2_history, time_axis):
    # inputs are 2D arrays containing the states of a liquid over time. Axis 0 is the liquid and axis 1 is time
    if liquid_1_history.shape != liquid_2_history.shape:
        raise Exception("Liquid histories must have same shape but have shape: ", liquid_1_history.shape, " and ", liquid_2_history.shape)
    elif len(liquid_1_history.shape) != 2:
        raise Exception("Liquid histories are not dimensional arrays, but are of dimension: ", len(liquid_1_history.shape))

    duration = liquid_1_history.shape[time_axis]

    # One of the cubes created above is then rotated such that the states align so that comparisons can be made between all states
    # To do: explain better

    difference_plane = np.zeros((duration, duration))
    for i0 in range(duration):
        for i1 in range(duration):
            print("percent complete: ", ((i0 * duration) + (i1+1))/(duration**2), end = '\r')
            difference_plane[i0,i1] = np.linalg.norm(liquid_1_history[:,i0] - liquid_2_history[:,i1])

    return difference_plane

def historywise_overlap(liquid_1_history, liquid_2_history):
    # inputs are 2D arrays containing the states of a liquid over time. Axis 0 is the liquid and axis 1 is time
    duration = liquid_1_history.shape[1]

    # One of the cubes created above is then rotated such that the states align so that comparisons can be made between all states
    # To do: explain better

    difference_plane = np.zeros((duration, duration))
    for i0 in range(duration):
        for i1 in range(duration):
            print("percent complete: ", ((i0 * duration) + (i1+1))/(duration**2), end = '\r')
            difference_plane[i0,i1] = np.sum((liquid_1_history[:,i0] + liquid_2_history[:,i1]) == 2)

    return difference_plane


def plot_distanceplane(liquid_1_history, liquid_2_history, color_range = None, colormap_size = None):

    z = historywise_separability(liquid_1_history, liquid_2_history)

    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show hight map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color_range != None:
        ax.set_zlim(color_range[0],color_range[1])
    ax.plot_surface(x, y, z)
    plt.title('z as 3d height map')
    plt.show()

    # show hight map in 2d
    if colormap_size != None:
        plt.figure(figsize = colormap_size)
    else:
        plt.figure()
    plt.title('z as 2d heat map')
    if color_range != None:
        p = plt.imshow(z, vmin = color_range[0], vmax = color_range[1])
    else:
        p = plt.imshow(z)
    plt.ylabel("Liquid 1 states")
    plt.xlabel("Liquid 2 states")
    plt.colorbar(p)
    plt.show()

def plot_overlapplane(liquid_1_history, liquid_2_history, color_range = None, colormap_size = None):

    z = historywise_overlap(liquid_1_history, liquid_2_history)

    x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

    # show hight map in 3d
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if color_range != None:
        ax.set_zlim(color_range[0],color_range[1])
    ax.plot_surface(x, y, z)
    plt.title('z as 3d height map')
    plt.show()

    # show hight map in 2d
    if colormap_size != None:
        plt.figure(figsize = colormap_size)
    else:
        plt.figure()
    plt.title('z as 2d heat map')
    if color_range != None:
        p = plt.imshow(z, vmin = color_range[0], vmax = color_range[1])
    else:
        p = plt.imshow(z)
    plt.ylabel("Liquid 1 states")
    plt.xlabel("Liquid 2 states")
    plt.colorbar(p)
    plt.show()
