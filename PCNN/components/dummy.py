import numpy as np

class Dummy_row:

    def __init__(self, **kwargs):
        self.neuron_arr = np.array(
            [
                np.random.random_sample(3),
                np.random.random_sample(3),
                np.random.random_sample(3),
            ],
            dtype=np.float16
        )

    def vals(self, y, x):
        val = self.neuron_arr[y, x] + 4 * (np.random.ranf() - 0.2)
        return  val
