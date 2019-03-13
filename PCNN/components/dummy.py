import numpy as np

class Dummy_row:

    def __init__(self, **kwargs):
        self.size_x = 3
        self.size_y = 3
        self.shape = (3, 3)
        self.arr = np.array(
            [
                np.random.random_sample(3),
                np.random.random_sample(3),
                np.random.random_sample(3),
            ],
            dtype=np.float16
        )

    def vals(self, y, x):
        val = self.arr[y, x] + 4 * (np.random.ranf() - 0.2)
        return val
