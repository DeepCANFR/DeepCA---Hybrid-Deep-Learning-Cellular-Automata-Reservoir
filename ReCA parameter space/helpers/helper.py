import numpy as np


def int_to_binary_string(int_number, size):
    binary_array = list(str(int(bin(int_number)[2:])))
    binary_array = list(map(int, binary_array))
    return left_pad_array(binary_array, size)


def binary_string_to_int(binary_string):
    s = map(str, binary_string)
    return int("".join(s), 2)

def left_pad_array(arr, size):
    array = np.zeros(size, dtype=int)
    array[-len(arr):] = arr
    return array


def pop_all_lists(list_of_lists):
    firsts = []
    for i in range(0, len(list_of_lists)):
        firsts.append(list_of_lists[i].pop(0))
    return firsts


def flatten_list_of_lists(lists):
    flat_list = []
    for sublist in lists:
        flat_list.extend(sublist)
    return flat_list
