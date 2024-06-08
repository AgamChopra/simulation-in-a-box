import torch.nn as nn

from neuron import Neuron

# Sensory Organs


class simple_eye_neuron(nn.Module):
    def __init__(self, weight):
        super(simple_eye_neuron, self).__init__()

    def forward(self, x):
        return


class complex_eye_neuron(nn.Module):
    def __init__(self, weight):
        super(complex_eye_neuron, self).__init__()

    def forward(self, x):
        return


class thermal_neuron(nn.Module):
    def __init__(self, weight):
        super(thermal_neuron, self).__init__()

    def forward(self, x):
        return


class photo_neuron(nn.Module):
    def __init__(self, weight):
        super(photo_neuron, self).__init__()

    def forward(self, x):
        return


class tactile_neuron(nn.Module):
    def __init__(self, weight):
        super(tactile_neuron, self).__init__()

    def forward(self, x):
        return


class clock_neuron(nn.Module):
    def __init__(self, weight):
        super(clock_neuron, self).__init__()

    def forward(self, x):
        return


class age_neuron(nn.Module):
    def __init__(self, weight):
        super(age_neuron, self).__init__()

    def forward(self, x):
        return


class hunger_neuron(nn.Module):
    def __init__(self, weight):
        super(age_neuron, self).__init__()

    def forward(self, x):
        return


# OUTPUT NEURONS
class velocity_adding_neuron(nn.Module):
    # Outputs % of max dv to be added to current velocity. tanh lets the nn decide the direction +ve 0-1 or -ve 0-1.
    def __init__(self, weight):
        super(velocity_adding_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Tanh()

    def forward(self, x):
        return self.nl(self.w * x)  # tensor of size [2] corrosponding to x,y.


class pheramon_output_neuron(nn.Module):
    def __init__(self, weight):
        super(pheramon_output_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return self.nl(self.w * x)


class orientation_neuron(nn.Module):
    # Angle the cell is pointing at. 0-360
    def __init__(self, weight):
        super(orientation_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return self.nl(self.w * x) * 360.


class kill_neuron(nn.Module):
    # Perform kill action in the direction the cell is pointing.
    def __init__(self, weight):
        super(kill_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False


class eat_neuron(nn.Module):
    # Perform eat action in the direction the cell is pointing.
    def __init__(self, weight):
        super(eat_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False


class reproduce_neuron(nn.Module):
    # send a reproduction signal to the cell to initiate reproduction.
    def __init__(self, weight):
        super(reproduce_neuron, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False


# HIDDEN NEURONS

class hidden_neuron(nn.Module):
    def __init__(self, weight):
        super(hidden_neuron, self).__init__()

    def forward(self, x):
        return


class hidden_recurrent_neuron(nn.Module):
    def __init__(self, weight):
        super(hidden_recurrent_neuron, self).__init__()

    def forward(self, x):
        return

# NEURON ID LIST

# GENE TO BRAIN GENERATOR

# HARDCODE

# INTERFACE
