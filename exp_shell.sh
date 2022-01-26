#!/bin/bash

dataset=("linear" "nonlinear")
method=("Trimming_change")
noise_rate=(0.01 0.05 0.1 0.2)

for a0 in "${dataset[@]}";do
        for a1 in "${method[@]}";do
                for a3 in "${noise_rate[@]}";do
                        python exp.py --dataset "${a0}" --method "${a1}" --noise_type instance --noise_rate "${a3}"
                done
        done
done