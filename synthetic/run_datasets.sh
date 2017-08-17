#!/bin/bash

#This script should be called with CUDA_VISIBLE_DEVICES
#already set. This script runs 1 of each gan model for
#1 of each dataset model

set -e

cvd=${CUDA_VISIBLE_DEVICES:?"Needs to be set"}
echo "DEVICES=$cvd"

#Sorry tqmd will produce some spastic output

for i in {1..30}
do
    echo "GPU "$CUDA_VISIBLE_DEVICES" Iter $i"

    python main.py --data_type=linear &
    python main.py --data_type=collider &
    python main.py --data_type=complete 

    #python main.py --data_type=network &
    #python main.py --data_type=network &
    #python main.py --data_type=network 

    #Make sure all finished
    echo "Sleeping"
    sleep 5m

done



echo "finshed fork_datasets.sh"

