# Copyright (c) 2019, Ahmed M. Alaa
# Licensed under the BSD 3-clause license (see LICENSE.txt)
import numpy as np
from scipy.cluster.vq import kmeans2
import torch
import torch.nn as nn
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import pyro.infer as infer
from pyro.contrib import autoname


# pyro.set_rng_seed(0)


class LinearT(nn.Module):
    """Linear transform and transpose"""

    def __init__(self, dim_in, dim_out):
        super(LinearT, self).__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, x):
        return self.linear(x).t()


class DeepGP(gp.parameterized.Parameterized):
    def __init__(self,
                 X,
                 y,
                 layer_dim=10,
                 layer_dim1=9,
                 inference="SVI",
                 mode="classification",
                 num_classes=2,
                 num_inducing=100
                 ):
        super(DeepGP, self).__init__()

        # Set all model attributes
        self.X = X
        self.y = y
        self.num_inducing = num_inducing
        self.num_classes = num_classes
        self.num_dim = self.X.shape[1]
        self.Xu = torch.from_numpy(kmeans2(self.X.numpy(), self.num_inducing, minit='points')[0])

        # set the inference algorithm to be used
        self.inference = inference

        # decide whether the model is to be usef for classification or regression
        self.mode = mode

        # handle erroneous settings for the model's parameters
        try:
            self.layer_dim = layer_dim
            self.layer_dim1 = layer_dim1

            if self.layer_dim < 2:
                raise ValueError("Bad inputs: number of intermediate dimensions must be greater than 2.")
        except ValueError as ve:
            print(ve)
            if self.layer_dim1 < 2:
                raise ValueError("Bad inputs: number of intermediate dimensions must be greater than 2.")
        except ValueError as ve1:
            print(ve1)

        # computes the weight for mean function of the first layer using a PCA transformation
        _, _, V = np.linalg.svd(self.X.numpy(), full_matrices=False)
        W = torch.from_numpy(V[:self.layer_dim, :])
        mean_fn = LinearT(self.num_dim, self.layer_dim)
        mean_fn.linear.weight.data = W
        mean_fn.linear.weight.requires_grad_(False)
        self.mean_fn = mean_fn

        X1 = self.mean_fn(self.X).t()
        _, _, V1 = np.linalg.svd(X1.numpy(), full_matrices=False)
        W1 = torch.from_numpy(V1[:self.layer_dim1, :])
        mean_fn1 = LinearT(self.layer_dim, self.layer_dim1)
        mean_fn1.linear.weight.data = W1
        mean_fn1.linear.weight.requires_grad_(False)
        self.mean_fn1 = mean_fn1

        # Initialize the first DGP layer
        self.layer_0 = gp.models.VariationalSparseGP(self.X, None,
                                                     gp.kernels.MQ(self.num_dim,
                                                                    variance=torch.tensor(2.),
                                                                    lengthscale=torch.tensor(2.)),
                                                     Xu=self.Xu,
                                                     likelihood=None,
                                                     mean_function=self.mean_fn,
                                                     latent_shape=torch.Size([self.layer_dim]))
        h = self.mean_fn(self.X).t()
        hu = self.mean_fn(self.Xu).t()

        self.layer_1 = gp.models.VariationalSparseGP(h,
                                                     None,
                                                     gp.kernels.MQ(self.layer_dim,
                                                                         variance=torch.tensor(2.),
                                                                         lengthscale=torch.tensor(2.)),
                                                     Xu=hu,
                                                     likelihood=None,
                                                     mean_function=self.mean_fn1,
                                                     latent_shape=torch.Size([self.layer_dim1]))
        b = self.mean_fn1(h).t()
        bu = self.mean_fn1(hu).t()

        self.layer_2 = gp.models.VariationalSparseGP(b,
                                                     self.y,
                                                     gp.kernels.MQ(self.layer_dim1,
                                                                   variance=torch.tensor(2.),
                                                                   lengthscale=torch.tensor(2.)),
                                                     Xu=bu,
                                                     likelihood=gp.likelihoods.MultiClass(num_classes=self.num_classes),
                                                     latent_shape=torch.Size([self.num_classes]))

        self.layer_0.u_scale_tril = self.layer_0.u_scale_tril * 1e-5
        # self.layer_0.set_constraint("u_scale_tril", torch.distributions.constraints.lower_cholesky)

    @autoname.name_count
    def model(self, X, y):
        self.layer_0.set_data(X, None)
        h_loc, h_var = self.layer_0.model()
        # approximate with a Monte Carlo sample (formula 15 of [1])
        h = dist.Normal(h_loc, h_var.sqrt())()

        self.layer_1.set_data(h.t(), None)
        i_loc, i_var = self.layer_1.model()
        # approximate with a Monte Carlo sample (formula 15 of [1])
        h1 = dist.Normal(i_loc, i_var.sqrt())()

        self.layer_2.set_data(h1.t(), y)
        self.layer_2.model()

    @autoname.name_count
    def guide(self, X, y):
        self.layer_0.guide()
        self.layer_1.guide()
        self.layer_2.guide()

    # make prediction
    def forward(self, X_new):
        # because prediction is stochastic (due to Monte Carlo sample of hidden layer),
        # we make 100 prediction and take the most common one (as in [4])
        pred = []
        num_MC_samples = 100
        for _ in range(num_MC_samples):
            h_loc, h_var = self.layer_0(X_new)
            h = dist.Normal(h_loc, h_var.sqrt())()

            i_loc, i_var = self.layer_1(h.t())
            h1 = dist.Normal(i_loc, i_var.sqrt())()

            f_loc, f_var = self.layer_2(h1.t())
            pred.append(f_loc.argmax(dim=0))

        return torch.stack(pred).mode(dim=0)[0]

    def train(self, num_epochs=5, num_iters=60, batch_size=1000, learning_rate=0.01):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = infer.TraceMeanField_ELBO().differentiable_loss
        for i in range(num_epochs):
            train(self.X, self.y, self, optimizer, loss_fn, batch_size, num_iters, i)


def train(X, y, gpmodule, optimizer, loss_fn, batch_size, num_iters, epoch):
    f = open("history.txt", 'a')
    for _ in range(num_iters):
        batch_indexes = np.random.choice(list(range(X.shape[0])), batch_size)
        data = X[batch_indexes, :]
        target = y[batch_indexes]
        data = data.reshape(-1, X.shape[1])
        optimizer.zero_grad()
        loss = loss_fn(gpmodule.model, gpmodule.guide, data, target)
        loss.backward()
        optimizer.step()
        print("Train Epoch: {:2d} \t[Iteration: {:2d}] \tLoss: {:.6f}".format(epoch, _, loss))
        strcontent = str("Train Epoch: {:2d} \t[Iteration: {:2d}] \tLoss: {:.6f}".format(epoch, _, loss))
        f.write(strcontent)
        f.write('\n')

