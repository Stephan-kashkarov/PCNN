import numpy as np
from PCNN.components.neuron import Neuron

def manually_fill_arr(size_x, size_y, obj):
    arr = []
    for y in range(size_y):
        arr_x = []
        for x in range(size_x):
            arr_x.append(obj(i=y, j=x))
        arr.append(arr_x)
    return np.array(arr)


class Row:
    def __init__(self, x, y, **kwargs):
        self.neurons = manually_fill_arr(x, y, Neuron, **kwargs)



def test_row():
    r = Row(9, 9)
    print(r.neurons)
