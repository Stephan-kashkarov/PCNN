import numpy as np
from matplotlib import pyplot as plt
from PCNN.components.row import Base, Dummy

class Grapher:
    
    def __init__(self, *args):
        self.vals = {str(key): [] for key in args}
        self.x = []
    
    def graph(self, **kwargs):
        for key, val in kwargs.items():
            self.vals[key].append(val)
        if self.x:
            self.x.append(self.x[-1] + 1)
        else:
            self.x.append(0)

    def show(self):
        plt.cla()
        for label, data in self.vals.items():
            plt.plot(self.x, data, label=label)
        plt.show()


class Neuron:

    def __init__(self, **kwargs):

        # Organisational Arguments
        keys = kwargs.keys()
        self.row = kwargs['row'] if 'row' in keys else Dummy_row()
        self.prev_row = kwargs['prev_row'] if 'prev_row' in keys else Dummy_row()

        # Scalar Arguments
        self.n = 0
        self.i = kwargs['i'].get()
        self.j = kwargs['j'].get()
        # Scalar parameters
        self.af = kwargs['af'] if 'af' in keys else np.float16(0.2)
        self.vf = kwargs['vf'] if 'vf' in keys else np.float16(0.1)
        self.al = kwargs['al'] if 'al' in keys else np.float16(0.5)
        self.vl = kwargs['vl'] if 'vl' in keys else np.float16(0.2)
        self.at = kwargs['at'] if 'at' in keys else np.float16(0.2)
        self.vt = kwargs['vt'] if 'vt' in keys else np.float16(6)
        self.bias = kwargs['bias'] if 'bias' in keys else np.float16(0.5)
        self.iterations = kwargs['it'] if 'it' in keys else np.float16(40)

        # Matrix parameters
        self.input_neurons = np.empty([3,3], dtype=np.int8)
        self.linker_weights = np.empty([3,3], dtype=np.float16)
        self.feeder_weights = np.empty([3,3], dtype=np.float16)

        # Matricies
        self.stimulus = np.float16(0)
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_act = np.float16(0)
        self.activation = np.float16(0)
        self.theta = np.float16(1)
        self.plot_bool = kwargs['plot'] if 'plot' in keys else False
        self.plotter = Grapher("stimulus", "feed", "link", "u_act", "theta", "activation")

    # Linking methods
    def populate(self):
        i = self.i
        j = self.j
        k = [i-1, i, i+1]
        l = [j-1, j, j+1]
        for x in k:
            for y in l:
                self.input_neurons[x, y] = self.prev_row.neuron_arr[y, x]
        self.input_neurons[i, j] = 0

    # Mathematical Method
    def calculate(self):
        self.populate()
        link_decay = np.exp(-(self.al), dtype=np.int8)
        feed_decay = np.exp(-(self.af), dtype=np.int8)
        theta_decay = np.exp(-(self.at), dtype=np.int8)
        weighted_feed = np.sum(np.multiply(self.input_neurons,self.feeder_weights))
        weighted_link = np.sum(np.multiply(self.input_neurons,self.linker_weights))
        self.feed = (feed_decay * self.feed) + (self.vf * weighted_feed) + self.stimulus
        self.link = (link_decay * self.link) + (self.vl * weighted_link)
        self.theta = (theta_decay*self.theta) + (self.vt * self.activation)
        self.u_act = self.feed * (1 + (self.bias * self.link))
        self.activation = 1 if self.u_act > self.theta else 0
        self.n += 1
        if self.plot_bool:
            self.plotter.graph(
                feed=self.feed,
                link=self.link,
                theta=self.theta,
                u_act=self.u_act,
                activation=self.activation,
            )
        return self.activation


    # Operational Method
    def iter(self, row, n):
        if self.n >= n:
            return self.activation
        else:
            while n >= self.n:
                yield self.calculate() # gives value of every calculation from self.n to n
            if self.plot_bool:
                self.plotter.show()
    
class Dummy_row:

    def __init__(self):
        self.neuron_arr = np.ones((3, 3), dtype=np.int8)

def test_run():
    n = Neuron(plot=True)
    r = Dummy_row()
    n.iter(r, n.iterations)
    print("done")

