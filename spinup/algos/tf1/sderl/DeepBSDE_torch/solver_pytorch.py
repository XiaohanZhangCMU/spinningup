import logging
import time
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam

MOMENTUM = 0.99
EPSILON = 1e-6
DELTA_CLIP = 50.0


class FeedForwardModel(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde, sess):
        self._config = config
        self._bsde = bsde
        self._sess = sess
        # make sure consistent with FBSDE equation
        self._dim = bsde.dim
        self._num_time_interval = bsde.num_time_interval
        self._total_time = bsde.total_time
        # ops for statistics update of batch normalization
        self._extra_train_ops = []


    def train(self):
        start_time = time.time()
        # to save iteration results
        training_history = []
        # for validation
        dw_valid, x_valid = self._bsde.sample(self._config.valid_size)
        # can still use batch norm of samples in the validation phase
        feed_dict_valid = {self._dw: dw_valid, self._x: x_valid, self._is_training: False}

        # initialization
        lr_lambda = lambda epoch: self._config.lr_values[0] if epoch < self._config.lr_boundaries[0] else self._config.lr_values[1]
        optimizer = torch.optim.Adam([self._y_init, self.z_init]+list(self._subnetwork.parameters()), lr=1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda) # what is initial learning rate?

        # begin sgd iteration
        for step in range(self._config.num_iterations+1):
            dw_train, x_train = self._bsde.sample(self._config.batch_size)
            loss = self.compute_loss(dw_train, x_train)

            if step % self._config.logging_frequency == 0:
                loss_val = loss.item()
                init = y_init.item()
                elapsed_time = time.time()-start_time+self._t_build
                training_history.append([step, loss, init, elapsed_time])
                if self._config.verbose:
                    logging.info("step: %5u,    loss: %.4e,   Y0: %.4e,  elapsed time %3u" % (
                        step, loss, init, elapsed_time))

            optimizer.zero_grad()
            loss.backward()
            # optimizer.step()
            scheduler.step()

        return np.array(training_history)


    def compute_loss(self, dw, x):
        start_time = time.time()
        time_stamp = np.arange(0, self._bsde.num_time_interval) * self._bsde.delta_t
        lo = torch.tensor([self._config.y_init_range[0]])
        hi = torch.tensor([self._config.y_init_range[1]])
        self._y_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)
        self._y_init = self._y_init.sample()

        print('minval = {0}, maxval = {1}'.format(self._config.y_init_range[0], self._config.y_init_range[1]))

        lo = -.1 * torch.ones(1,self._dim)
        hi =  .1 * torch.ones(1,self._dim)
        z_init = torch.distributions.uniform.Uniform(lo, hi, validate_args=None)
        z_init = z_init.sample()

        all_one_vec = torch.ones([self._dw[0],1])
        y = all_one_vec * self._y_init
        z = all_one_vec * self.z_init

        for t in range(0, self._num_time_interval-1):
            y = y - self._bsde.delta_t * (
                self._bsde.f_tf(time_stamp[t], x[:, :, t], y, z)
            ) + torch.sum(z * dw[:, :, t], 1, keep_dims=True)
            z = self._subnetwork(x[:, :, t + 1], str(t + 1)) / self._dim
        # terminal time
        y = y - self._bsde.delta_t * self._bsde.f_tf(
            time_stamp[-1], x[:, :, -2], y, z
        ) + torch.sum(z * dw[:, :, -1], 1, keep_dims=True)
        delta = y - self._bsde.g_tf(self._total_time, self._x[:, :, -1])
        # use linear approximation outside the clipped range
        loss = torch.mean(torch.where(torch.abs(delta) < DELTA_CLIP, torch.pow(delta,2), 2*DELTA_CLIP * torch.abs(delta) - DELTA_CLIP**2))
        self._t_build = time.time()-start_time

        return loss


    def _subnetwork(nn.Module):
        def __init__(self):
            layers = []
            activation = nn.ReLU
            output_activation = nn.Identity

            for j in range(1, len(self._config.num_hiddens)):
                act = activation if j < len(sizes)-2 else output_activation
                layers += [nn.BatchNorm1d(sizes[j], affine=True), nn.Linear(sizes[j], sizes[j+1]), act()]
            self.net = nn.Sequential(*layers)

            def init_weights(m):
                if type(m) == nn.Linear:
                    torch.nn.init.xavier_uniform(m.weight)
                    #m.bias.data.fill_(0.01)
            self.net.apply(init_weights)

        def forward(self, x):
            z = self.net(x)
            return z

