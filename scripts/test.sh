#!/bin/bash

# Runs the tests for the ablation studies

MODEL=google/gemma-3-1b-it;
DATASET=abisee/cnn_dailymail;
RESULTS_DIRECTORY=$(realpath $(dirname "$BASH_SCRIPT"))/models/bash;
NAME="Ablation $(date +%s)"
EVALUATION_DIRECTORY="~/$NAME"

while getopts "m:d:r:n:e:h" opt; do
	case $opt in
		m) MODEL="$OPTARG" ;;
		d) DATASET="$OPTARG" ;;
		r) RESULTS_DIRECTORY="$OPTARG" ;;
		n) NAME="$OPTARG" ;;
		e) EVALUATION_DIRECTORY="$OPTARG" ;;
		h) printf """ablation is a script designed to automatically prune models to be used for the ablation study.

Arguments:
	-m: The base model that is being pruned.
	-d: The dataset that is being used to prune and evaluate.
	-r: The parent directory where the resulting pruned models are stored.
	-n: The name of the test to be put in the database (Should be unique so only models created during this run are selected)
	-e: The directory where the evaluation results will be written.
	-h: Displays this help message.
		"""; exit 0;;
		\?) echo "Invalid option: $opt"; exit 1;;
	esac
done
shift $((OPTIND - 1))

echo "MODEL=$MODEL"
echo "DATASET=$DATASET"
echo "RESULTS_DIRECTORY=$RESULTS_DIRECTORY"
echo "NAME=$NAME"
echo "EVALUATION_DIRECTORY=$EVALUATION_DIRECTORY"