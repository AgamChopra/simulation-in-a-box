import torch.nn as nn

from neuron import Neuron

# Sensory Organs
class eye_(nn.Module):
    # Passive Energy Consumption(0-1): 0.01
    def __init__(self, weight):
        super(eye_, self).__init__()

    def forward(self, x):
        return


class thermal_(nn.Module):
    # Passive Energy Consumption(0-1): 0.001
    def __init__(self, weight):
        super(thermal_, self).__init__()

    def forward(self, x):
        return


class photo_(nn.Module):
    # Passive Energy Consumption(0-1): 0.001
    def __init__(self, weight):
        super(photo_, self).__init__()

    def forward(self, x):
        return


class tactile_(nn.Module):
    # Passive Energy Consumption(0-1): 0.001
    def __init__(self, weight):
        super(tactile_, self).__init__()

    def forward(self, x):
        return


# OUTPUT NEURONS
class velocity_add_(nn.Module):
    # Passive Energy Consumption(0-1): 0.005
    # Active Energy Consumption(0-1): max(0.05, 0.5 * mass * velo ** 2)

    # Outputs % of max dv to be added to current velocity. tanh lets the nn decide the direction +ve 0-1 or -ve 0-1.
    def __init__(self, weight):
        super(velocity_add_, self).__init__()
        self.w = weight
        self.nl = nn.Tanh()

    def forward(self, x):
        return self.nl(self.w * x)  # tensor of size [2] corrosponding to x,y.


class orientation_(nn.Module):
    # Passive Energy Consumption(0-1): 0.001
    # Active Energy Consumption(0-1): max(0.01, mass * abs(coeff_drag))
        
    # Angle the cell is pointing at. 0-360
    def __init__(self, weight):
        super(orientation_, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return self.nl(self.w * x) * 360.


class kill_(nn.Module): # HIGH RISK HIGH REWARD, Takes A LOT of energy
    # Passive Energy Consumption(0-1): 0.0
    # Active Energy Consumption(0-1): 0.3

    # Perform kill action in the direction the cell is pointing.
    def __init__(self, weight):
        super(kill_, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False


class eat_(nn.Module):
    # Passive Energy Consumption(0-1): 0.0
    # Active Energy Consumption(0-1): 0.1

    # Perform eat action in the direction the cell is pointing.
    def __init__(self, weight):
        super(eat_, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False


#change this: make it so its sexual or asexual...
class reproduce_(nn.Module):
    # Passive Energy Consumption(0-1): 0.01
    # Active Energy Consumption(0-1): 0.75

    # send a reproduction signal to the cell to initiate reproduction.
    def __init__(self, weight):
        super(reproduce_, self).__init__()
        self.w = weight
        self.nl = nn.Sigmoid()

    def forward(self, x):
        return True if self.nl(self.w * x) > 0.5 else False
