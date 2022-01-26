#!/bin/bash
model_name=("resnet50" "dnet169" "dnet121" "resnet18")

python base_line.py --dataset covid_ct --method CE --noise_type symmetric --noise_rate 0.0 --model_name dnet121 --optim sgd
for a0 in "${model_name[@]}";do
        python base_line.py --dataset covid_ct --method CE --noise_type symmetric --noise_rate 0.0 --model_name "${a0}" --optim adam 
done

# model_name=("dnet169" "dnet121" "resnet18" "resnet50")
# optim=("sgd" "sgd_w" "adam")

# for a0 in "${model_name[@]}";do
#         for a1 in "${optim[@]}";do
#                 python best_acc_loop.py --dataset covid_ct --method CE --noise_type symmetric --noise_rate 0.0 --model_name "${a0}" --optim "${a1}" 
#         done
# done

