#!/bin/sh

data_folder="data_wsm"

if [ ! -d ./data/${data_folder} ] ; then
    scp -r xiaohan.zhang@xiaohanzha-wsm:~/Codes/spinningup/data ./data/${data_folder}
fi

python -m spinup.run plot /Users/xiaohan.zhang/Downloads/Graphics_Scratch/toplot
#/Users/xiaohan.zhang/Codes/spinningup/data/${data_folder}/vpg
