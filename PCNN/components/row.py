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

def map_arr_iterate(arr):
    for val in arr.reshape(-1):
        yield val.iterate()


class Row:
    def __init__(self, y, x, **kwargs):
        prev_row = kwargs.get("prev_row")
        self.prev_row = prev_row if prev_row else Base_row(shape=(y, x), dtype=np.float16)
        self.neurons = manually_fill_arr(x, y, Neuron, prev_row=self.prev_row, row=self)
        self.values = np.zeros((y, x), dtype=np.float16)
        if kwargs.get("plot"):
            coords = kwargs.get("plot_coords")
            if coords:
                y, x = coords
            else:
                y = np.random.randint(0, y, dtype=np.uint16)
                x = np.random.randint(0, x, dtype=np.uint16)
            self.neurons[y, x].plot_bool = True
            # print(self.neurons[y, x].plot_bool)

    def iterate(self):
        self.values = np.array(list(map_arr_iterate(self.neurons)), dtype=np.float16)

class Base_row:

    def __init__(self, **kwargs):
        arr = kwargs.get('arr')
        dtype = kwargs.get("dtype", np.float16)
        if arr:
            self.arr = np.array(arr, dtype=dtype)
        else:
            y, x = kwargs.get('shape')
            self.arr = np.array(
                [np.random.random_sample(x) for h in range(y)],
                dtype=dtype
            )

    def vals(self, y, x):
        return self.arr[y, x]

def test_row():
    r = Row(9, 9, plot=True)
    # print(r.neurons)
