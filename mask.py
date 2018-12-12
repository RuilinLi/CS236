import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

# ------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class Generator(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=True):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        #assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [2*nin] + hidden_sizes + [nout]
        for h0,h1 in zip(hs, hs[1:]):
            self.net.extend([
                    MaskedLinear(h0, h1),
                    nn.ReLU(),
                ])
        self.net.pop() # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0 # for cycling through num_masks orderings

        self.m = {}
        self.update_masks() # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

        # ml2 = MaskedLinear(in_features=2*self.nin, out_features=self.nin, bias=True)
        # tmp = np.arange(2*self.nin)
        # tmp2 = np.arange(self.nin)
        # mask = (tmp[:,None] == tmp2[None,:]) + (tmp[:,None] == tmp2[None,:] + self.nin)
        # ml2.set_mask(mask)
        # self.net2 = ml2

    def update_masks(self):
        if self.m and self.num_masks == 1: return # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin, 2*self.nin) # if self.natural_ordering else rng.permutation(np.arange(self.nin, 2*self.nin))
        for l in range(L):
            self.m[l] = rng.randint(self.m[l-1].min(), 2*self.nin-1, size=self.hidden_sizes[l])

        # construct the mask matrices
        self.m[-1] = np.arange(2*self.nin)
        masks = [self.m[l-1][:,None] <= self.m[l][None,:] for l in range(L)]
        self.m[-1] = np.arange(self.nin, 2*self.nin)
        masks.append(self.m[L-1][:,None] < self.m[-1][None,:])
        #masks[0] = np.concatenate((np.ones_like(masks[0]), masks[0]), 0)

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]]*k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l,m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x, z):
        x_reverse = torch.flip(x, [1])
        y1 = self.net(torch.cat((x, z), 1))
        y2 = self.net(torch.cat((x_reverse, z), 1)) #Maybe using a different net?
        return y1 + torch.flip(y2,[1])

class Discriminator(nn.Module):
    def __init__(self, x_dim, hidden_sizes):
        super(Discriminator, self).__init__()

        """
        self.model1 = nn.Sequential(
        nn.Linear(2*x_dim, h_dim),
        nn.ReLU(),
        nn.Linear(h_dim, 1)
        )
        """

        self.model1 = []
        hs = [2*x_dim] + hidden_sizes + [1]
        for h0,h1 in zip(hs, hs[1:]):
            self.model1.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
        self.model1.pop() # pop the last ReLU for the output layer
        self.model1 = nn.Sequential(*self.model1)

        #model2 is used to determine swap
        """
        # self.model2 = nn.Sequential(
        # nn.Linear(2*x_dim, h_dim),
        # nn.ReLU(),
        # nn.Linear(h_dim, x_dim),
        # #nn.Hardtanh(min_val=0, max_val=1)
        # nn.Sigmoid()
        # )
        """

        self.model2 = []
        hs = [2*x_dim] + hidden_sizes + [x_dim]
        for h0,h1 in zip(hs, hs[1:]):
            self.model2.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
        self.model2.pop() # pop the last ReLU for the output layer
        self.model2.append(nn.Sigmoid())
        self.model2 = nn.Sequential(*self.model2)

    def forward(self, x, x_tilde, swap):
        if swap:
            """
            Here I will use a 'soft swap'. That is, generate
            an interpolation parameter t from (0,1) and replace x with
            t*x_tilde + (1-t)*x. Similar operation is done to x_tilde
            """
            t = self.model2(torch.cat((x, x_tilde), 1))
            x_swaped = t*x_tilde + (1-t)*x
            x_tilde_swaped = t*x + (1-t)*x_tilde
            return self.model1(torch.cat((x_swaped, x_tilde_swaped), 1))
        else:
            return self.model1(torch.cat((x,x_tilde), 1))
def train():

    x_dim, h_dim, z_dim, n = 20, 10, 20, 100

    generator = Generator(x_dim, [h_dim], x_dim)
    discriminator = Discriminator(x_dim, h_dim)

    x = torch.randn(n, x_dim)

    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=1e-4)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=1e-4)

    ncritic = 5
    Tensor = torch.FloatTensor
    for t in range(500):
        z = Variable(Tensor(np.random.normal(0, 1, (n, z_dim))))
        x_tilde = generator.forward(x,z).detach()

        optimizer_D.zero_grad()
        #loss_D = -torch.mean(discriminator(x, x_tilde, 0)) + torch.mean(discriminator(x, x_tilde, 1))
        loss_D = -torch.mean(torch.pow(discriminator(x, x_tilde, 0) - discriminator(x, x_tilde, 1),2))
        loss_D.backward()
        optimizer_D.step()
        # Clip weights of discriminator
        for p in discriminator.parameters():
            p.data.clamp_(-0.1, 0.1)

        # Train the generator every n_critic iterations
        if t % ncritic == 0:
            optimizer_G.zero_grad()

            x_tilde = generator.forward(x,z)
            # Adversarial loss
            #loss_G = torch.mean(discriminator(x, x_tilde, 0)) - torch.mean(discriminator(x, x_tilde, 1))
            loss_G = torch.mean(torch.pow(discriminator(x, x_tilde, 0) - discriminator(x, x_tilde, 1),2))
            loss_G.backward()
            optimizer_G.step()
    #z = Variable(Tensor(np.random.normal(0, 1, (n, z_dim))))
