import numpy as np
import matplotlib.pyplot as plt

class PCNN:
    def __init__(self, **kwargs):
        self.n = 0
        self.input_shape = kwargs.get('input_shape', (3, 3))
        self.input_size = self.input_shape[0] * self.input_shape[1]

        # Neuron matrixes
        self.neurons = np.array(
            [Neuron(**kwargs.get("neuron_params", {})) for x in range(self.input_size)],
            dtype=np.object
        ).reshape(self.input_shape)

        # wow this is messy, sorry!
        for index, neuron in np.ndenumerate(self.neurons):
            neuron.i = index[1]
            neuron.j = index[0]

        self.activation = [np.empty(self.input_shape),]

        # Graphing
        self.history = np.array([
            {
                'feed': [],
                'link': [],
                'u_activation': [],
                'theta': [],
            } for neuron in range(self.input_size)
        ]).reshape(self.input_shape)

    def __call__(self, x, iterations=10):
        assert x.shape == self.input_shape
        
        for neuron in self.neurons.flatten():
            his = neuron(x)
            self.record(neuron.i, neuron.j, his)
        self.activation.append(np.empty(self.input_shape))
        yield self.activation[-1]
        for i in range(iterations - 1):
            for neuron in self.neurons.flatten():
                his = neuron(self.activation[-1])
                self.record(neuron.i, neuron.j, his)
            self.activation.append(np.empty(self.input_shape))
            yield self.activation[-1]

    def record(self, x, y, his):
        self.activation[-1][y, x] = his[0]
        for key in self.history[y, x].keys():
            self.history[y, x][key].append(his[1].pop())
        

class Neuron:
    def __init__(self, **kwargs):
        
        # Constant Arguments
        self.i = kwargs.get('i')
        self.j = kwargs.get('j')
        self.kernal_shape = [3, 3]
        self.kernal_size = self.kernal_shape[0] * self.kernal_shape[1]


        # Constant Parameters
        self.yx = kwargs.get('yf', np.float16(0.7))
        self.af = kwargs.get('af', np.float16(0.2))
        self.vf = kwargs.get('vf', np.float16(0.1))
        self.al = kwargs.get('al', np.float16(0.2))
        self.vl = kwargs.get('vl', np.float16(0.1))
        self.at = kwargs.get('at', np.float16(0.2))
        self.vt = kwargs.get('vt', np.float16(10))
        self.bias = kwargs.get('bias', np.float16(0.2))

        # Output Parameters
        self.feed = np.float16(0)
        self.link = np.float16(0)
        self.u_activation = np.float16(0)
        self.theta = np.float16(1)
        self.activation = np.float16(0)

        # Weight Tensors
        self.linker_weights = np.random.normal(size=(self.kernal_shape[0], self.kernal_shape[1]))
        self.feeder_weights = np.random.normal(size=(self.kernal_shape[0], self.kernal_shape[1]))

        self.linker_weights[int(self.kernal_shape[1]/2), int(self.kernal_shape[0]/2)] = 0
        self.feeder_weights[int(self.kernal_shape[1]/2), int(self.kernal_shape[0]/2)] = 0


    def __call__(self, data):
        
        inputs = []
        for y in range(self.j - 1, self.j + 2):
            for x in range(self.i - 1, self.i+2):
                try:
                    inputs.append(data[y, x])
                except:
                    inputs.append(0)
        layer_inputs = np.array(inputs).reshape(3, 3)
        
        feed_weights = self.vf * np.sum(np.multiply(self.feeder_weights, layer_inputs))
        link_weights = self.vl * np.sum(np.multiply(self.linker_weights, layer_inputs))

        self.feed = (np.exp(-self.af) * self.feed) + feed_weights
        self.link = (np.exp(-self.al) * self.link) + link_weights + data[self.j, self.i]
        self.u_activation = self.feed * (1 + (self.link * self.bias))
        self.theta = (np.exp(-self.at)*self.theta) + self.vt * self.activation
        self.activation = 1 if self.u_activation > self.theta else 0

        return [self.activation, [self.feed, self.link, self.u_activation, self.theta]]
