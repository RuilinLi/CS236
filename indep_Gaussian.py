import random
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal

import toy_model

seed = 14
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_default_dtype(torch.float32)

N=100 # num of samples
D=2 # num of dimension
M=1 # num of relevant
Tensor = torch.FloatTensor # Tensor = torch.DoubleTensor # for float64


sigma = 2. * Categorical(Tensor([.5,.5])).sample((M,)) - 1.
sigma = sigma.type(Tensor)
alpha = torch.linspace(2,5,10) # 10 different values of alpha
inx = torch.arange(D,dtype=torch.float64).multinomial(M) # random sample inx for relevant features




def set_up(i):
    '''
    Get data
    :param i: The index for alpha value to control the strength of influence of X has on Y
    :return:X,Y
    '''
    #X = Normal(0., 1. / D).sample((N, D))
    X = MultivariateNormal(torch.zeros(D), torch.eye(D) / D).rsample((N,))
    mu = torch.sum(X[:, inx] * sigma * alpha[i], dim=1)
    Y = Normal(mu, 1.).sample()

    return X,Y

def train_indep_gaussian(X,Y,h_dim=10,z_dim=10,iters=500):
    x_dim, n = X.shape[-1], X.shape[0]
    generator = toy_model.Generator(x_dim, h_dim, z_dim)
    discriminator = toy_model.Discriminator(x_dim, h_dim)
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    ncritic = 5

    G_loss = []
    D_loss = []
    for i in range(iters):
        for _ in range(ncritic):
            z = Variable(Tensor(np.random.normal(0, 1, (n, z_dim))))
            x_tilde = generator.model(torch.cat((X, z), 1)).detach()

            optimizer_D.zero_grad()
            loss_D = -torch.mean(discriminator(X, x_tilde, 0)) + torch.mean(discriminator(X, x_tilde, 1))
            loss_D.backward()
            optimizer_D.step()
            D_loss.append(loss_D.detach())

            for p in discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)

        optimizer_G.zero_grad()
        x_tilde = generator.model(torch.cat((X, z), 1))
        loss_G = torch.mean(discriminator(X, x_tilde, 0)) - torch.mean(discriminator(X, x_tilde, 1))
        loss_G.backward()
        optimizer_G.step()
        G_loss.append(loss_G.detach())







