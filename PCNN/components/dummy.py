import numpy as np

class Dummy_row:

    def __init__(self):
        self.neuron_arr = np.array(
            [
                np.random.random_sample(3),
                np.random.random_sample(3),
                np.random.random_sample(3),
            ],
            dtype=np.float16
        )
        print(f"neuron array is {self.neuron_arr}")

    def vals(self, x, y):
        return self.neuron_arr[x, y] + 4 * (np.random.ranf() - 0.2)


class Dummy:
    def iter(self, *args):
        pass
