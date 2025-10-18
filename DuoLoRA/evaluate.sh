#!/bin/bash
eval_dir=$1

echo "eval_dir: $eval_dir"

python evaluate_both.py --eval_dir $eval_dir
