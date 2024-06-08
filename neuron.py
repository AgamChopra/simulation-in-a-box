"""
Created on Fri Jun  7 14:01:12 2024

@author: agam
"""

import torch
from torch import nn
from random import randint
from matplotlib import pyplot as plt


class Neuron(nn.Module):
    def __init__(self, weights, charge_time=5, self_firing=False,
                 plasticity=0.1):
        super(Neuron, self).__init__()
        self.weights = nn.Parameter(weights, requires_grad=False)
        self.weights /= (((self.weights)**2).sum()**0.5 + 1E-6)
        self.tau = torch.tensor(0.)
        self.charge_time = charge_time  # in terms of frames
        self.charge_timer = randint(0, charge_time)  # frame delay
        self.self_firing = self_firing
        self.sigmoid = nn.Sigmoid()
        self.plasticity = plasticity
        self.plastic_factor = .5

    def switch_on_fast_learning(self, lam=2):
        self.plastic_factor = lam

    def switch_off_learning(self):
        self.plastic_factor = 0.

    def reset_learning(self):
        self.plastic_factor = 0.5

    def update_timer(self):
        if self.charge_timer > 0:
            self.charge_timer -= 1

    def reset_timer(self):
        self.charge_timer = self.charge_time

    def update_weights(self, x):
        kappa = torch.where(x > 0.5, self.plastic_factor, -self.plastic_factor)
        tick = -self.tau * torch.abs(self.weights)
        self.weights += 1E-1 * kappa * tick
        self.weights /= (((self.weights)**2).sum()**0.5 + 1E-6)

    def forward(self, x):
        y = x @ self.weights
        y = self.sigmoid(y)
        y = torch.where(y > 0.6, y, torch.tensor(0.0))

        if self.charge_timer == 0:
            if self.self_firing:
                y = torch.clip(y + torch.rand_like(y), 0.6, 1)
            self.reset_timer()
            self.tau = -0.1 * y
            if self.plasticity:
                self.update_weights(x)
            return y
        elif self.charge_timer == self.charge_time:
            self.update_timer()
            return self.tau
        else:
            self.update_timer()
            return torch.tensor(0.0)


if __name__ == '__main__':
    w = torch.rand((4), requires_grad=False)
    neuron = Neuron(w, self_firing=False, plasticity=True)
    response = []
    synapse = [[], [], [], []]
    [synapse[i].append(neuron.weights[i].item()) for i in range(len(synapse))]

    for cycle in range(500):
        if cycle < 200:
            neuron.switch_on_fast_learning()
        elif cycle < 450:
            neuron.reset_learning()
        else:
            neuron.switch_off_learning()
        x = torch.rand_like(w, requires_grad=False)
        response.append(neuron(x).item())
        [synapse[i].append(neuron.weights[i].item())
         for i in range(len(synapse))]

    plt.plot(response)
    plt.title('Neuron Output')
    plt.show()

    [plt.plot(synapse[i]) for i in range(len(synapse))]
    plt.title('Input Synapse Connection Strength ie: weight')
    plt.show()
