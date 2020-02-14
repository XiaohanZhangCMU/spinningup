#!/bin/sh

cd ~/Codes/spinningup

python -m spinup.run ppo_pytorch --env Walker2d-v2 --exp_name walker --act torch.nn.ELU
