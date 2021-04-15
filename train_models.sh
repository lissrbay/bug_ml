#!/bin/sh
truncate -s 0 conda_info.txt
conda info >> conda_info.txt
path_to_conda=$(grep -o 'active env location : .*$' conda_info.txt)
path_to_conda=($path_to_conda)
path_to_conda=${path_to_conda[4]}
path_to_python=${path_to_conda}'/envs/bug_ml/bin/python'

path_to_reports=$
$path_to_python baseline.py ${path_to_reports}
$path_to_python train_model.py all