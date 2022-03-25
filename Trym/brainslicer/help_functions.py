import string
import random
'''
Help functions
'''


class UniqueIdDictCreator(object):
    def __init__(self, ID_length):
        self.ID_length = ID_length
        self.existing_IDs = []

    def create_unique_ID_dict(self, dictionary):
        dictionary = dictionary.copy()

        letters = string.ascii_lowercase
        ID = "".join(random.choice(letters) for i in range(30))
        while ID in self.existing_IDs:
            ID = "".join(random.choice(letters) for i in range(30))
        dictionary["ID"] = ID
        self.existing_IDs.append(ID)
        return dictionary


def remove_neg_values(array, mean, SD):
    population_size = array.shape
    negative_values = array <= 0
    replacement_values = ncp.random.uniform(
        mean - SD, mean + SD, population_size)
    array *= negative_values == 0
    array += replacement_values*negative_values
