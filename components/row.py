import numpy as np

class Row:
    pass


class Base:
    def __init__(self, activation):
        self.activation = activation

    def iter(self, *args):
        return self.activation


class Dummy:
    """A class to prevent recursive errors"""

    def iter(self, *args):
        pass
