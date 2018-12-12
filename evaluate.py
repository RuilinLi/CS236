import numpy as np
from sklearn.linear_model import LassoCV
import simulator
import torch
import pdb

class Eval():

    def __init__(self, x, x_tilde):
        """
        Here x is assumed to be a PyTorch Tensor
        but y is a numpy array
        """
        self.xt = x_tilde.numpy()
        self.x = x.numpy()
        self.p = self.x.shape[1]

    def simulate_y(self, sigma, nsig):
        """
        self.y = y(x) according to sparse linear model
        sigma is the variance of the nosie and
        nsig is the number of significant predictors
        """
        beta = np.zeros(self.p)
        sigind = np.random.choice(np.arange(self.p),nsig,replace=False)
        beta[sigind] = 1
        self.y = np.matmul(self.x, beta) + sigma*np.random.randn(self.x.shape[0])
        self.truth = sigind


    def LCD(self):
        """
        Compute the Lasso coefficient difference statistic
        """
        #lasso = linear_model.Lasso(alpha=reg)
        #lasso.fit(np.concatenate((self.x,self.xt), axis = 1), self.y)
        reg = LassoCV(cv=5, random_state=0).fit(np.concatenate((self.x,self.xt), axis = 1), self.y)
        abs_beta = abs(reg.coef_)
        return abs_beta[0:self.p] - abs_beta[self.p:]

    def selection(self, q, stat, offset=True):


        """
        Here q is the desired FDR rate.
        stat is obtained from knockoff procedure, and
        it needs to have a symmetric distribution under
        the null hypothesis
        """
        """
        p = self.p
        idx = np.argsort(stat)
        sorted_stat = stat[idx]
        zero_idx = np.searchsorted(sorted_stat, 0)
        for i in range(min(p-zero_idx, zero_idx)):
            t = []
            W = sorted_stat[zero_idx+i]
            numerator = np.searchsorted(sorted_stat, W, side='right')
            denominator = p - i - zero_idx
            if (numerator/denominator) <= q:
                t.append(W)
            W = abs(sorted_stat[zero_idx-1-i])
            numerator = zero_idx-i
            denominator = np.searchsorted(sorted_stat, W, side='right')
            if (numerator/denominator) <= q:
                t.append(W)

            if (len(t) > 0):
                t = min(t)
                break

        big_idx = np.searchsorted(sorted_stat, t)
        return idx[big_idx:]
        """
        idx = np.argsort(np.abs(stat))
        for i in idx:
            total = sum(stat>=abs(stat[i]))
            false = sum(stat <= -abs(stat[i]))
            #print("total {} false {}".format(total, false))
            if offset:
                fdr = (false+1.0) / total
            else:
                fdr = false / total
            if fdr <= q:
                break
            #print(i, fdr)
        if fdr > q:
            return np.array([])
        #pdb.set_trace()
        return np.where(stat>= abs(stat[i]))[0]

    def FDR(self, rejected):
            """
            truth is the index for variables where the null hypothesis is true
            """
            #The naming is a bit confusing here. self.truth is supposed
            #to be the indices where the alternative hypothesis is true
            #but truth here is the indecies where the null is true
            truth = set(range(self.p)).difference(set(self.truth))
            if len(rejected) == 0:
                return 0
            rejected = set(rejected)
            truth = set(truth)
            return len(truth.intersection(rejected))/len(rejected)

    def power(self, rejected):
            """
            h1 is the index for variables where the alternative hypothesis is true
            """
            rejected = set(rejected)
            h1 = set(self.truth)
            return len(h1.intersection(rejected))/len(h1)
