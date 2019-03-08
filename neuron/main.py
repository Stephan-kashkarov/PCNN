default_scalar_variables = {
    "af": 0.2,
    "vf": 0.1,
    "al": 0.5,
    "vl": 0.2,
    "at": 0.2,
    "vt": 6,
    "bias": 0.5,
    "iterations": 40,
}



class Neuron:
    def __init__(self, i, j, af, vf, al, vl, at, vt, bias, iterations):

        # Scalar Arguments
        self.n = 0
        self.i = i
        self.j = j

        # Scalar parameters
        self.af = af if af else 0.2
        self.vf = vf if af else 0.1
        self.al = al if af else 0.5
        self.vl = vl if af else 0.2
        self.at = at if af else 0.2
        self.vt = vt if af else 6
        self.bias = bias if bias else 0.5
        self.iterations = iterations if iterations else 40

        # Matrix parameters
        self.feeder_cells = []
        self.feeder_inputs = {}
        self.linker_cells = []
        self.linker_inputs = {}

        # Matricies
        self.feed = 0
        self.link = 0
        self.u_act = 0
        self.e_act = 0
        self.threshold = 0

    # Linking methods
    def populate_feeder(self):
        pass

    def populate_linker(self):
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