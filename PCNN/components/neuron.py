import numpy as np
from matplotlib import pyplot as plt

from PCNN.components.dummy import Dummy_row


class Grapher:
    
    def __init__(self, *args):
        self.vals = {str(key): [] for key in args}
        self.x = []
    
    def graph(self, **kwargs):
        for key, val in kwargs.items():
            self.vals[key].append(val)
        if len(self.x):
            self.x.append(self.x[-1] + 1)
        else:
            self.x.append(1)

    def show(self):
        plt.cla()
        for label, data in self.vals.items():
            if len(data) != self.x[-1]: # error prevention
                continue
            plt.plot(self.x, data, "-+", label=label)
        plt.legend()
        plt.show()


class Neuron:

    def __init__(self, **kwargs):

        # Organisational Arguments
        self.row = kwargs.get('row')
        self.prev_row = kwargs.get('prev_row')
        
        if self.row == None:
            self.row = Dummy_row()
        if self.prev_row == None:
            self.prev_row = Dummy_row()

        print(f"prevous row is:\n{self.prev_row.arr}")

        # Scalar Arguments
        self.n = 0
        self.i = kwargs.get('i')
        self.j = kwargs.get('j')

        # Scalar parameters
        self.yx = kwargs.get('yf', np.float16(0.7))
        self.af = kwargs.get('af', np.float16(0.2))
        self.vf = kwargs.get('vf', np.float16(0.1))
        self.al = kwargs.get('al', np.float16(0.2))
        self.vl = kwargs.get('vl', np.float16(0.1))
        self.at = kwargs.get('at', np.float16(0.2))
        self.vt = kwargs.get('vt', np.float16(8))
        self.bias = kwargs.get('bias', np.float16(0.2))
        self.iterations = kwargs.get('it', 40)
        # Matrix parameters
        self.input_neurons = np.empty(
            (3, 3),
            dtype=np.float16
        )
        self.linker_weights = kwargs.get("linker_weights", np.array(
            [[np.random.random_sample(3)] for h in range(3)],
            dtype=np.float16
        ))
        self.feeder_weights = kwargs.get("feeder_weights", np.array(
            [[np.random.random_sample(3)] for h in range(3)],
            dtype=np.float16
        ))
        print(self.linker_weights)
        print(self.feeder_weights)

        # Matricies
        self.stimulus = self.prev_row.vals(self.i, self.j)
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_act = np.float16(0)
        self.activation_internal = np.float16(0)
        self.theta = np.float16(1)
        self.plot_bool = kwargs.get('plot', False)
        self.plotter = Grapher(
            "feed", 
            "link", 
            "u_act", 
            "theta", 
            "activation",
        )

    # Linking methods
    def populate(self):
        i = self.i
        j = self.j
        k = [i-1, i, i+1]
        l = [j-1, j, j+1]
        for x in k:
            for y in l:
                if x in range(self.prev_row.size_x) and y in range(self.prev_row.size_y):
                    val = self.prev_row.vals(y, x)
                else:
                    val = 0
                np.put(self.input_neurons, [y, x], val)
        self.stimulus = self.input_neurons[i, j]
        np.put(self.input_neurons, [i, j], 0)
        # print(self.input_neurons)
        # print(self.prev_row.vals(x, y))

    # Mathematical Method
    def calculate(self):
        print(f"Calculating neuron: {self.i} {self.j}:")
        self.populate()
        link_decay = np.exp(-(self.al))
        feed_decay = np.exp(-(self.af))
        theta_decay = np.exp(-(self.at))
        weighted_feed = np.sum(
            np.multiply(self.input_neurons,self.feeder_weights)
        )
        weighted_link = np.sum(np.multiply(self.input_neurons,self.linker_weights))
        print(f"Feed = ({feed_decay} x {self.feed}) + ({self.vf} x {weighted_feed}) + {self.stimulus}", end=" ")
        self.feed = (feed_decay * self.feed) + (self.vf * weighted_feed) + self.stimulus
        print(f"= {self.feed}")
        print(f"Link = ({link_decay} x {self.link}) + ({self.vl} x {weighted_link})", end=" ")
        self.link = (link_decay * self.link) + (self.vl * weighted_link)
        print(f"= {self.link}")
        self.u_act = self.feed * (1 + (self.bias * self.link))
        print(f"U = {self.feed} x (1 + ({self.bias} x {self.link})) = {self.u_act}")
        print(f"Theta = ({theta_decay} x {self.theta}) + ({self.vt} x {self.activation_internal})", end=" ")
        self.theta = (theta_decay * self.theta) + (self.vt * self.activation_internal)
        print(f"= {self.theta}")
        self.activation_internal = 1 if self.u_act > self.theta else 0
        self.n += 1
        activation = (1 / (1 + np.exp(self.yx * (self.u_act - self.theta))))
        if self.plot_bool:
            self.plotter.graph(
                feed=self.feed,
                link=self.link,
                theta=self.theta,
                u_act=self.u_act,
                activation=activation,
            )
        return activation


    # Operational Method
    def pulse(self, n):
        if self.n >= n:
            if self.plot_bool:
                self.plotter.show()
            return self.activation_internal
        else:
            self.calculate()
            return self.pulse(n) # gives value of every calculation from self.n to n
    

def test_neuron():
    while True:
        n = Neuron(plot=True, i=1, j=1)
        n.pulse(n.iterations)
    print("done")
