import numpy as np
import torch
import matplotlib.pyplot as plt


class Simulator():
    def __init__(self):
        """
        Simulating randon pairs (x,y)
        where x has distribution
        """
        pass

    def GMM(self, mu, sd, k):
        """
        Generate Gaussian mixture random variables
        with mean mu and variance var, k uniform
        mixture components.
        mu: dims, number of mixture components
        sd: scalar standard deviation
        """
        pass

    def AR(self, sigma, T, n, c=0, coef=0.5):
        """
        Simulate an Autoregressive process AR(1)
        X[t] = c + coef * X[t-1] + sigma*Z[t], with X[0] = 0
        """
        X = np.zeros((n, T))
        for i in range(1, T):
            X[:,i] += c + coef * X[:,i-1] + sigma*np.random.randn(n)
        return torch.FloatTensor(X)

if __name__ == '__main__':
    a = Simulator()
    X = a.AR(1, 1500, 15, coef=0.8)
    plt.plot(X[0,:])
    plt.show()
