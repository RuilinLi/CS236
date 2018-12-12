import mask
import simulator
import evaluate
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pdb


if __name__ == '__main__':
    seed = 14
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float32)
    Tensor = torch.FloatTensor # Tensor = torch.DoubleTensor # for float64

    D = 5
    N = 500
    noise_sigma = 0.5
    x_sigma = 1
    x_dim, h_dim, z_dim, n = D, 5, D, N
    k, q = 1, 0.1
    offset = True
    niter = 10000

    sim = simulator.Simulator()
    X = sim.AR(x_sigma, D, N)

    generator = mask.Generator(x_dim, [h_dim+5, h_dim, h_dim], z_dim)
    discriminator = mask.Discriminator(x_dim, [h_dim])
    #optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    #optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    #optimizer_G = torch.optim.SGD(generator.parameters(), lr = 0.01, momentum=0.9)
    #optimizer_D = torch.optim.SGD(discriminator.parameters(), lr = 0.01, momentum=0.9)
    ncritic = 10

    G_loss = []
    D_loss = []
    for i in tqdm(range(niter)):
        for _ in range(ncritic):
            z = Variable(Tensor(np.random.normal(0, 1, (n, z_dim))))
            x_tilde = generator.forward(X, z).detach()

            optimizer_D.zero_grad()
            loss_D = -torch.mean(torch.pow(discriminator(X, x_tilde, 0) - discriminator(X, x_tilde, 1),2))
            loss_D.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-0.1, 0.1)
        optimizer_G.zero_grad()
        x_tilde = generator.forward(X, z)
        loss_G = torch.mean(torch.pow(discriminator(X, x_tilde, 0) - discriminator(X, x_tilde, 1),2))
        loss_G.backward()
        optimizer_G.step()
        G_loss.append(loss_G.detach())
        D_loss.append(loss_D.detach())

    E = evaluate.Eval(X, x_tilde.detach())
    E.simulate_y(noise_sigma, k)
    lasso_difference = E.LCD()
    rejected = E.selection(q, lasso_difference, offset)
    print(lasso_difference)
    print(rejected)
    print(E.truth)
    print(E.FDR(rejected))
    print(E.power(rejected))
    cov_matrix = np.cov(np.concatenate((X.numpy(), x_tilde.detach().numpy()), 1),rowvar=False)
    np.set_printoptions(precision=4, suppress=True)
    print(np.diagonal(cov_matrix))
