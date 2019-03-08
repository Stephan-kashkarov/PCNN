import numpy as np

class Neuron:
    def __init__(self, **kwargs):

        # Scalar Arguments
        self.n = 0
        self.i = kwargs['i']
        self.j = kwargs['j']

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
    def populate(self, row):
        link = row.neurons()
        i = self.i
        j = self.j
        np.put(self.input_neurons, range(0, 9), [
            [x.run() for x in [link[i-1][j-1], link[i+0][j-1], link[i+1][j-1],]],
            [x.run() for x in [link[i-1][j+0], Dummy()       , link[i+1][j+0],]],
            [x.run() for x in [link[i-1][j+1], link[i+0][j+1], link[i+1][j+1],]],
        ])

    # Mathematical Methods
    def get_f(self, stimulus):
        pass

    def get_l(self):
        pass

    def get_u(self):
        pass

    def get_y(self):
        pass

    def get_t(self):
        pass


class Dummy:
    """A class to prevent recursive errors"""
    def run(self):
        return 0
