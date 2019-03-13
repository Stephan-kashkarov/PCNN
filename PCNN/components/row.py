import numpy as np
from PCNN.components.neuron import Neuron

def manually_fill_arr(size_x, size_y, obj, **kwargs):
    arr = []
    for y in range(size_y):
        arr_x = []
        for x in range(size_x):
            arr_x.append(obj(i=y, j=x, **kwargs))
        arr.append(arr_x)
    return np.array(arr, dtype=object)

def map_arr_iterate(arr, graph=False):
    a = arr.reshape(-1)
    for val in a:
        print(val.i, val.j)
        yield val.pulse(val.n + 1, graph=graph)


class Row:
    def __init__(self, y, x, **kwargs):
        self.prev_row = kwargs.get(
            "prev_row", Base_row(
                shape=(y, x),
                dtype=np.float16
            )
        )
        self.neurons = manually_fill_arr(x, y, Neuron, prev_row=self.prev_row, row=self, shape=(y, x))
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

    def iterate(self, graph=False):
        self.values = np.array(list(map_arr_iterate(self.neurons)), dtype=np.float16)
        return self.values
    
    def vals(self, i, j):
        return self.values[i, j]

class Base_row:

    def __init__(self, **kwargs):
        arr = kwargs.get('arr')
        dtype = kwargs.get("dtype", np.float16)
        if arr:
            self.arr = np.array(arr, dtype=dtype)
            self.size_y, self.size_x = self.arr.shape
        else:
            self.size_y, self.size_x = kwargs.get('shape')
            self.arr = np.array(
                [np.random.random_sample(self.size_x)
                 for h in range(self.size_y)],
                dtype=dtype
            )

    def vals(self, y, x):
        return self.arr[y, x]

def test_row():
    br = Base_row(shape=(9, 9))
    print(f"Base image\n{br.arr}")
    r = Row(9, 9, plot=True)
    for i in range(40):
        r.iterate()
    print(r.iterate())
    # print(r.neurons)
