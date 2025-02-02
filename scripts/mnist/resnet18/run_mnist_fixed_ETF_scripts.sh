#!/bin/bash

cd nc

## =====================================================================================
## Experiments for mnist dataset and AGD with Cross Entropy Loss
## =====================================================================================

## Run for multiple different seeds
for seed in {1..3};
do
    python3 train.py --gpu_id 0 --storage data/ --uid mnist-fixed-etf-agd/seed-$seed --dataset mnist \
        --model resnet18 --optimiser AGD --no-bias --batch_size 256 --force --use_cudnn --epochs 200 \
        --loss CrossEntropy --temperature 5 --log_interval 10 --ETF_fc --seed $seed

    python3 validate_NC.py --gpu_id 0 --storage data/ --dataset mnist --model resnet18 --batch_size 256 --no-bias --ETF_fc \
        --uid mnist-fixed-etf-agd/seed-$seed --epochs 200 --nc_metrics train --log_interval 5 --seed $seed

    python3 validate_NC.py --gpu_id 0 --storage data/ --dataset mnist --model resnet18 --batch_size 256 --no-bias --ETF_fc \
        --uid mnist-fixed-etf-agd/seed-$seed --epochs 200 --nc_metrics test --log_interval 5 --seed $seed

    python3 ETF_distance.py --gpu_id 0 --storage data/ --uid mnist-fixed-etf-agd/seed-$seed --force --use_cudnn \
        --epochs 200 --num_classes 10 --dataset mnist --model resnet18 --nc_metrics train --log_interval 5 --ETF_fc --seed $seed

    python3 ETF_distance.py --gpu_id 0 --storage data/ --uid mnist-fixed-etf-agd/seed-$seed --force --use_cudnn \
        --epochs 200 --num_classes 10 --dataset mnist --model resnet18 --nc_metrics test --log_interval 5 --ETF_fc --seed $seed
done

python3 seed_statistics.py --storage data/ --uid mnist-fixed-etf-agd  --dataset mnist --model resnet18 \
    --nc_metrics train --log_interval 5 --epochs 200 --ETF_fc --ref_ETF 1

python3 seed_statistics.py --storage data/ --uid mnist-fixed-etf-agd --dataset mnist --model resnet18 \
    --nc_metrics test --log_interval 5 --epochs 200 --ETF_fc --ref_ETF 1
