#!/bin/sh

data_folder="data_wsm"

if [ ! -d ./data/${data_folder} ] ; then
    scp -r xiaohan.zhang@xiaohanzha-wsm:~/Codes/spinningup/data ./data/${data_folder}
fi

python -m spinup.run plot /Users/xiaohan.zhang/Codes/spinningup/data/${data_folder}/dvpg_LunarLander-v2_lunarlander-v2_eps0-01_pi0-0003/

python -m spinup.run plot /Users/xiaohan.zhang/Codes/spinningup/data/${data_folder}/dvpg_LunarLander-v2_lunarlander-v2_eps0-01_pi0-03/
