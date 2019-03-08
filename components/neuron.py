import numpy as np

class Neuron:
    def __init__(self, **kargs):

        # Scalar Arguments
        self.n = 0
        self.i = i
        self.j = j

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
        self.feeder_cells = np.matrix(
            [None, None, None], [None, None, None], [None, None, None])
        self.feeder_inputs = np.matrix([0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.feeder_weights = np.matrix([0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.linker_cells = np.matrix(
            [None, None, None], [None, None, None], [None, None, None])
        self.linker_inputs = np.matrix([0, 0, 0], [0, 0, 0], [0, 0, 0])
        self.linker_weights = np.matrix([0, 0, 0], [0, 0, 0], [0, 0, 0])

        # Matricies
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_act = np.float16(0)
        self.e_act = np.float16(0)
        self.theta = np.float16(0)

    # Linking methods
    def auto_populate_feeder(self, row):
        pass

    def auto_populate_linker(self, row):
        pass

    def man_populate_feeder(self, *args):
        pass

    def man_populate_linker(self, *args):
        pass

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
