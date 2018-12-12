import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import Normal
from torch.distributions import Categorical


import toy_model


seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)

N=200 # num of samples
D=2 # num of dimension
M=1 # num of relevant
Tensor = torch.FloatTensor # Tensor = torch.DoubleTensor # for float64


class GMM():
    '''
    e.g.)

    m1 = MultivariateNormal(torch.zeros(2) + 10.,torch.eye(2) * 1)
    m2 = MultivariateNormal(torch.zeros(2) - 10.,torch.eye(2) * 1)
    mix = GMM(cat = Tensor([.5,.5]), components = [m1,m2], dim=2)
    >>> mix._log_prob(Tensor([[1.,1.],[0.,0.]])) # compute the log probability
    >>> mix._sample(10) # sample

    '''
    def __init__(self,cat,components,dim):
        '''

        :param cat: 1D tensor: categorical distribution for mixture (e.g.) cat = torch.Tensor([.5,.5])
        :param components: a list where each component is a gaussian distribution
        '''
        super(GMM, self).__init__()

        if torch.sum(cat) != 1.:
            raise ValueError("Cat must sum up to one")

        if len(cat) != len(components):
            raise ValueError("Number of components must match size of categorical tensor")

        self._cat = cat
        self._components = components
        self.num_comp = len(components)
        self.dim = dim

    def _sample(self,num_samples):
        b = Categorical(self._cat).sample((num_samples,))
        b = b.view([-1, 1])
        mask = torch.zeros(num_samples, self.num_comp)
        mask.scatter_(1, b, 1)

        sample = torch.zeros((num_samples, self.num_comp, self.dim))
        for i in range(self.num_comp):
            sample[:, i, :] = self._components[i].sample((num_samples,))

        return torch.sum(mask[:,:,None] * sample,dim=1)

    def _log_prob(self,x):
        if len(x.shape) == 1:
            num_sample = 1
        else:
            num_sample = x.shape[0]
        log_prob = torch.zeros((num_sample,self.num_comp))
        for i in range(self.num_comp):
            log_prob[:,i] = (torch.log(self._cat[i]) + self._components[i].log_prob(x))

        return torch.logsumexp(log_prob, dim=1)
