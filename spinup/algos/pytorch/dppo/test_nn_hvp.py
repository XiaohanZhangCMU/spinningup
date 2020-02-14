import numpy as np
import torch
import gym
from torch.optim import Adam
import time
import spinup.algos.pytorch.dppo.core as core
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
action_space = gym.spaces.Discrete(4)

ac_kwargs=dict()

ac = core.MLPActorCritic(observation_space, action_space, **ac_kwargs)
sync_params(ac)

obs1 = 3* torch.rand(4000,8)
act1 = 3* torch.rand(4000)
#
#obs = torch.load('obs.pt')
#act = torch.load('act.pt')
#
#print(obs1.shape, obs.shape)
#print(act1.shape, act.shape)
#
#print(obs1.min(), obs1.max())
#print(obs1.min(), obs.max())


pi_optimizer = Adam(ac.pi.parameters(), lr=0.001)

pi_optimizer.zero_grad()

pi, logp = ac.pi(obs1, act1)

loss_pi = logp.mean()

loss_pi.backward()

pi_optimizer.step()

print(loss_pi)


