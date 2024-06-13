import torch
from torch import nn

from neuron import Neuron


class Cluster(nn.Module):
    def __init__(self, weights=[], charge_time=[], self_firing=[],
                 plasticity=[]):
        super(Cluster, self).__init__()
        neurons = nn.ModuleDict({
            'input_neuron_1': Neuron(weights[0], charge_time[0], False, plasticity[0]),
            'input_neuron_2': Neuron(weights[1], charge_time[1], False, plasticity[1]),
            'clock_neuron': Neuron([0.,0.,0.], charge_time[2], True, False),
            'hidden_neuron_1': Neuron(weights[3], charge_time[3], self_firing[3], plasticity[3]),
            'hidden_neuron_1': Neuron(weights[4], charge_time[4], self_firing[4], plasticity[4]),
            
            })


class Brain(nn.Module):
    def __init__(self, weights, charge_time=5, self_firing=False,
                 plasticity=False):
        super(Brain, self).__init__()