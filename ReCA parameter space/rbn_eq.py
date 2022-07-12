# Elementary CA includes the CA that exclude computation from one or more of the cells.
# this is a short program to find the 3 sets that do this for left central and right nodes.
# this reduces the ECA to even smaller computational universes

import numpy as np
import helpers.helper as helper

lookup = {}
for i in range(0, 256):
    binary = helper.int_to_binary_string(i, 8)
    print(i, binary)
    eq = []
    eq.append(helper.binary_string_to_int([binary[0],
                                           binary[2], binary[1], binary[3],
                                           binary[4], binary[6], binary[5],
                                           binary[7]]))

    eq.append(helper.binary_string_to_int([binary[0],
                                           binary[1], binary[4], binary[5],
                                           binary[2], binary[3], binary[6],
                                           binary[7]]))

    eq.append(helper.binary_string_to_int([binary[0],
                                           binary[4], binary[1], binary[5],
                                           binary[2], binary[6], binary[3],
                                           binary[7]]))

    eq.append(helper.binary_string_to_int([binary[0],
                                           binary[2], binary[4], binary[6],
                                           binary[1], binary[3], binary[5],
                                           binary[7]]))

    eq.append(helper.binary_string_to_int([binary[0],
                                           binary[4], binary[2], binary[6],
                                           binary[1], binary[5], binary[3],
                                           binary[7]]))
    lookup[i] = eq


# print(len(lookup.keys()))
count = 0
for key in lookup.keys():
    if key <= min(lookup[key]):
        eq = list(dict.fromkeys(lookup[key]))
        if key in eq:
            eq.remove(key)
        eq = [key] + eq
        print(eq)
        # print(key, "&",  eq, "\\\\")
        # count += 1

# print(count)

#
# for i in range(0, 16):
#     binary = helper.int_to_binary_string(i, 4)
#     print(i, binary)
