#!/bin/sh

cd ~/Codes/spinningup

#python -m spinup.run dppo_pytorch --env Walker2d-v2 --exp_name walker --act torch.nn.ELU
python -m spinup.run dppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999

