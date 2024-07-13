#!/bin/bash
env=$1
algo=$2

echo env name is $env
echo algo name is $algo

shift
shift
echo $@

if [ -z "$1" ]; then
  python benchmark/train.py --algo $algo --env $env --verbose 0 -P --eval-freq -1 --tensorboard-log ./tblog/ --track --wandb-project-name rlzoo3_${env}
else
  python benchmark/train.py --algo $algo --env $env --verbose 0 -P --eval-freq -1 --tensorboard-log ./tblog/ --track --wandb-project-name rlzoo3_${env} -params $@ -tags $@
fi
