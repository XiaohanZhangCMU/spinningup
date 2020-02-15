import numpy as np
import torch
import gym
from torch.optim import Adam
import time
import spinup.algos.pytorch.dppo.core as core
from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

# Test Pytorch layer schema for mannually interfering parameters` gradient

observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
action_space = gym.spaces.Discrete(4)

ac_kwargs=dict()

ac = core.MLPActorCritic(observation_space, action_space, **ac_kwargs)
sync_params(ac)

#obs = 3* torch.rand(1,8)
#act = 3* torch.rand(1)
#torch.save(obs, 'obs.pt')
#torch.save(act, 'act.pt')
obs = torch.load('obs.pt')
act = torch.load('act.pt')

pi_optimizer = Adam(ac.pi.parameters(), lr=0.001)
pi_optimizer.zero_grad()

pi, logp = ac.pi(obs, act)
loss_pi = logp.mean()

print('Before Backward Propagation')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad)
        break
    break

# Compute gradients d loss_pi/dw
loss_pi.backward()

print('Before Maunipulating Gradients')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad[0][0])
        break
    break

# Modify gradients with customized values
if 1:
    for l in ac.pi.logits_net:
        for x in l.parameters():
            old_grad = x.grad
            new_grad = 0*torch.ones(old_grad.shape)
            x.grad = new_grad

# Update weights according to Adam schema
pi_optimizer.step()

# Print out updated weights
print('After One Step of Minimization')
for l in ac.pi.logits_net:
    for x in l.parameters():
        print(x.data[0][0])
        print(x.grad[0][0])
        break
    break
