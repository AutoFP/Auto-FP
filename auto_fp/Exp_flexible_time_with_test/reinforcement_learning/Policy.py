import torch
import torch.nn as nn

from copy import deepcopy

class Policy(nn.Module):
    def __init__(self, max_length, search_space):
        super(Policy, self).__init__()
        self.max_length   = max_length
        self.search_space = deepcopy(search_space)
        self.arch_parameters = nn.Parameter( 1e-3*torch.randn(self.max_length, len(search_space)) )
    
    def generate_arch(self, actions):
        temp_actions = deepcopy(actions)
        result = []
        for item in temp_actions:
            if (item == 7):
                continue
            result.append(item)
        return result


    def forward(self):
        alphas  = nn.functional.softmax(self.arch_parameters, dim=-1)
        return alphas


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""
    def __init__(self, momentum):
        self._numerator   = 0
        self._denominator = 0
        self._momentum    = momentum

    def update(self, value):
        self._numerator = self._momentum * self._numerator + (1 - self._momentum) * value
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator