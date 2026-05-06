#!/bin/bash

NOW="1778011282";
MODEL="google/gemma-3-1b-it";
DATASET="abisee/cnn_dailymail";
RUN_NAME="Ablation $NOW";
EVALUATION_DIRECTORY="$HOME/evaluation/ablation_$NOW";
clear;
time python3.14 -m washLTH -f="$EVALUATION_DIRECTORY" evaluate --dataset="$DATASET" --baseline="$MODEL" --max-evaluations=15 --mask-filter=magnitude --sql-filter="WHERE \"name\" REGEXP '$RUN_NAME \((Unt|T)rained\) \(\d+\.?\d*%\)'" --ablation --baseline-as-target || curl -d "Pipeline failed with exit code $?!" "$NTFY_LINK";
