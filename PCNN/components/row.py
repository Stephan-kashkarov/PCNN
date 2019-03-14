import numpy as np
from PIL import Image
from PCNN.components.neuron import Neuron

def manually_fill_arr(size_x, size_y, obj, **kwargs):
    arr = []
    for y in range(size_y):
        arr_x = []
        for x in range(size_x):
            arr_x.append(obj(i=y, j=x, **kwargs))
        arr.append(arr_x)
    arr = np.array(arr, dtype=object)
    return arr

def pulse_arr(arr):
    a = arr.reshape(-1)
    for val in a:
        yield val.pulse(val.n + 1)


class Row:
    def __init__(self, y, x, **kwargs):
        self.prev_row = kwargs.get(
            "prev_row", Base_row(
                shape=(y, x),
                dtype=np.float16
            )
        )
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
        self.x = x
        self.y = y

    def iterate(self, n=1):
        while n:
            self.values = np.array(
                list(pulse_arr(self.neurons)),
                dtype=np.float16
            ).reshape(self.y, self.x)
            n -= 1
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
    # while True:
    n = Neuron()
    br = Base_row(shape=(9, 9))
    r = Row(9, 9, plot=False, prev_row=br)
    r.iterate(20)
    print(f'{"-"*28} Second last {"-"*28}')
    print(r.values)
    r.iterate()
    print(f'{"-"*30} Results {"-"*30}')
    print(r.values)
