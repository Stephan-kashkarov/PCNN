import numpy as np
from PCNN.components.row import Base, Dummy

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
        self.stimulus = np.float16(0)
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_act = np.float16(0)
        self.e_act = np.float16(0)
        self.theta = np.float16(1)

    # Linking methods
    def populate(self):
        i = self.i
        j = self.j
        k = [i-1, i, i+1]
        l = [j-1, j, j+1]
        for x in k:
            for y in l:
                self.input_neurons[x, y] = self.prev_row.neuron_dict[y, x]
        self.input_neurons[i, j] = 0

    # Monitoring Methods
    def graph_state(self):
        pass

    def show_graph(self):
        pass

    # Mathematical Methods
    def get_f(self):
        decay = np.exp(-(self.af), dtype=np.int8)
        weighted_feed = np.sum(
            np.multiply(
                self.input_neurons,
                self.feeder_weights,
            )
        )
        self.feed = (decay * self.feed) + (self.vf * weighted_feed) + self.stimulus
        

    def get_l(self):
        decay = np.exp(-(self.al), dtype=np.int8)
        weighted_link = np.sum(
            np.multiply(
                self.input_neurons,
                self.linker_weights,
            )
        )
        self.link = (decay * self.link) + (self.vl * weighted_link)

    def get_u(self):
        self.get_f()
        self.get_l()
        self.u_act = self.feed * (1 + (self.bias * self.link))

    def get_t(self):
        decay = np.exp(-(self.at), dtype=np.int8)
        self.theta = (decay*self.theta) + (self.vt * self.e_act)

    def get_y(self):
        self.n += 1
        self.get_u()
        self.get_t()
        self.e_act = 1 if self.u_act > self.theta else 0
        self.graph_state()


    # Operational Method
    def iter(self, row, n):
        if n > self.n:
            self.get_y()
            if n == self.iter:
                self.show_graph()
                return None
        return self.e_act
    
    def test_run(self):
        pass

