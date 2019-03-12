import numpy as np
from PCNN.components.neuron import Neuron

def manually_fill_arr(size_x, size_y, obj, **kwargs):
    arr = []
    for y in range(size_y):
        arr_x = []
        for x in range(size_x):
            arr_x.append(obj(i=y, j=x, **kwargs))
        arr.append(arr_x)
    return np.array(arr)

def map_arr(arr, func):
    pass


class Row:
    def __init__(self, y, x, **kwargs):
        prev_row = kwargs.get("prev_row")
        self.prev_row = prev_row if prev_row else np.empty((y, x), dtype=np.float16)
        self.neurons = manually_fill_arr(x, y, Neuron, prev_row=self.prev_row)
        self.values = np.zeros((y, x), dtype=np.float16)

    def iterate(self):
        pass

class Base_row:

    def __init__(self, arr):
        self.arr = np.array(arr, dtype=np.float16)

    def vals(self, y, x):
        return self.arr[y, x]

def test_row():
    r = Row(9, 9)
    print(r.neurons)
