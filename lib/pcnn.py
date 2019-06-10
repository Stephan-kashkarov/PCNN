import numpy as np
import matplotlib.pyplot as plt

class PCNN:
    def __init__(self, **kwargs):
        
        self.input_shape = kwargs.get('input_shape', (3, 3))
        self.input_size = self.input_shape[0] * self.input_shape[1]

        # Neuron matrixes
        self.neurons = np.array(
            [Neuron(**kwargs.get("neuron_params", {})) for x in range(self.input_size)],
            dtype=np.object
        ).reshape(self.input_shape)

        # Graphing
        self.history = {
            'theta': [],
            'u_activation': [],
            'link': [],
            'feed': [],
            'activation': [],
        }

    def __call__(self, x, iterations=10):
        print(self.input_shape, x.shape)
        # assert x.shape == self.input_shape
        
        for neuron in self.neurons.flatten():
            his = neuron(x)
            yield his[0]
            self.record(his)
        for i in range(iterations - 1):
            for neuron in self.neurons.flatten():
                his = neuron(self.neurons.activation)
                yield his[0]
                self.record(his)

    def record(self, his):
        for key in self.history.keys():
            self.history[key].append(his.pop())

class Neuron:
    def __init__(self, **kwargs):
        
        # Constant Arguments
        self.i = kwargs.get('i')
        self.j = kwargs.get('j')
        self.kernal_shape = kwargs.get('kernal_shape', (3, 3))
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


    def __call__(self, x):
        
        layer_inputs = x[
            self.j - self.kernal_shape[1]/2:self.j + (self.kernal_shape[1]/2)+1,
            self.i - self.kernal_shape[0]/2:self.i + (self.kernal_shape[0]/2)+1,
        ].reshape(self.kernal_shape)
        
        feed_weights = self.vf * np.matmul(self.feeder_weights, layer_inputs)
        link_weights = self.vl * np.matmul(self.linker_weights, layer_inputs)

        self.feed = (np.exp(-self.af) * self.feed) + feed_weights
        self.link = (np.exp(-self.al) * self.link) + link_weights + x[self.j, self.i]
        self.u_activation = self.feed * (1 + (self.link) * self.bias)
        self.theta = (np.exp(-self.at)*self.theta) + self.vt * self.activation
        self.activation = 1 if self.u_activation > self.theta else 0

        return [self.activation, self.feed, self.link, self.u_activation, self.theta]
