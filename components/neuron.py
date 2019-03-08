import numpy as np

class Neuron:
    def __init__(self, **kwargs):


        # Organisational Arguments
        self.row = kwargs['row'] if kwargs['row'] else Dummy()
        self.prev_row = kwargs['prev_row'] if kwargs['prev_row'] else Dummy()

        # Scalar Arguments
        self.n = 0
        self.i = kwargs['i'] if kwargs['i'] else Dummy()
        self.j = kwargs['j'] if kwargs['j'] else Dummy()

        # Scalar parameters
        self.af = kwargs['af'] if kwargs['af'] else np.float16(0.2)
        self.vf = kwargs['vf'] if kwargs['vf'] else np.float16(0.1)
        self.al = kwargs['al'] if kwargs['al'] else np.float16(0.5)
        self.vl = kwargs['vl'] if kwargs['vl'] else np.float16(0.2)
        self.at = kwargs['at'] if kwargs['at'] else np.float16(0.2)
        self.vt = kwargs['vt'] if kwargs['vt'] else np.float16(6)
        self.bias = kwargs['bias'] if kwargs['bias'] else np.float16(0.5)
        self.iterations = kwargs['it'] if kwargs['it'] else np.float16(40)

        # Matrix parameters
        self.cells = []
        self.input_neurons = np.empty([3,3], dtype=np.int8)
        self.linker_weights = np.empty([3,3], dtype=np.float16)
        self.feeder_weights = np.empty([3,3], dtype=np.float16)

        # Matricies
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_act = np.float16(0)
        self.e_act = np.float16(0)
        self.theta = np.float16(1)

    # Linking methods
    def populate(self):
        link = self.prev_row.neurons()
        i = self.i
        j = self.j
        np.put(self.input_neurons, range(0, 9), [
            [x.iter(self.prev_row, self.n - 1) for x in [link[i-1, j-1], link[i+0, j-1], link[i+1, j-1],]],
            [x.iter(self.prev_row, self.n - 1) for x in [link[i-1, j+0], Dummy()       , link[i+1, j+0],]],
            [x.iter(self.prev_row, self.n - 1) for x in [link[i-1, j+1], link[i+0, j+1], link[i+1, j+1],]],
        ])

    # Mathematical Methods
    def get_f(self, stimulus):
        decay = np.exp(-(self.af), dtype=np.int8)
        weighted_feed = np.sum(np.multiply(self.input_neurons, self.feeder_weights))
        

    def get_l(self):
        pass

    def get_u(self):
        pass

    def get_t(self):
        pass

    def get_y(self):
        self.n += 1
        pass

    def iter(self, row, n):
        if n > self.n:

        else:
            return self.e_act

class Base:
    def __init__(self, activation):
        self.activation = activation
    
    def iter(self, *args):
        return self.activation


class Dummy:
    """A class to prevent recursive errors"""
    def iter(self, *args):
        pass
