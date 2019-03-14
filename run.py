# from PCNN.components.network import PCNN
# from PCNN.components.settings import init_scalar_variables
from PCNN.components.visualisation import visualise_random, visualise_input
from PCNN.components.neuron import test_neuron
from PCNN.components.row import test_row
import random

visualise_input(f"frames/seq_{random.randint(1, 2000):06d}.jpg", 50, 100)
