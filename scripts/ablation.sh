#!/bin/bash

# Runs the tests for the ablation studies
# TODO: Figure out if I can automatically redirect stdout and stderr to a file and screen with script

NOW=$(date +%s)
MODEL=google/gemma-3-1b-it;
DATASET=abisee/cnn_dailymail;
RESULTS_DIRECTORY="$(realpath $(dirname "$BASH_SCRIPT"))/models/ablation_$NOW";
RUN_NAME="Ablation $NOW"
EVALUATION_DIRECTORY="$HOME/evaluation/ablation_$NOW"

while getopts "m:d:r:n:e:h" opt; do
	case $opt in
		m) MODEL="$OPTARG" ;;
		d) DATASET="$OPTARG" ;;
		r) RESULTS_DIRECTORY="$OPTARG" ;;
		n) RUN_NAME="$OPTARG" ;;
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

# mkdir -p "$RESULTS_DIRECTORY/{trained,untrained}";
mkdir -p "$EVALUATION_DIRECTORY"
printf "Pruned models will be placed in: $RESULTS_DIRECTORY.\nEvaluation results will be placed in: $EVALUATION_DIRECTORY.\nName will be $RUN_NAME."

PRUNING_PERCENTS=$(seq 0.05 0.05 0.975);
NUM_ITERATIONS=$(echo $PRUNING_PERCENTS | wc -w);

for PRUNE in $(echo $PRUNING_PERCENTS); do
	for TRAIN_BOOL in true false; do
		curl \
			-d "Started pruning." \
			-H "Title: Pruning parameter $PRUNE (Training: $TRAIN_BOOL)" \
			-H "Tags: microscope" \
			"$NTFY_LINK";

		PRUNE_PERCENT=$(echo "$PRUNE * 100" | bc | sed -E 's/\.0{1,}$//')
		if [[ "$TRAIN_BOOL" == "true" ]]; then
			python3.14 -m washLTH -f="$RESULTS_DIRECTORY/trained/$PRUNE" train --prune="$PRUNE" --model="$MODEL" --dataset="$DATASET" --name="$RUN_NAME (Trained) ($PRUNE_PERCENT%)";
			EXIT_CODE=$?;
		else
			python3.14 -m washLTH -f="$RESULTS_DIRECTORY/untrained/$PRUNE" train --prune="$PRUNE" --model="$MODEL" --name="$RUN_NAME (Untrained) ($PRUNE_PERCENT%)";
			EXIT_CODE=$?;
		fi

		if [[ "$EXIT_CODE" -eq 0 ]]; then
		   curl \
			-d "Pruning parameter $PRUNE finished successfully!" \
			-H "Title: Pruning parameter $PRUNE (Training: $TRAIN_BOOL)" \
			-H "Tags: tada" \
			"$NTFY_LINK";
		else
		   curl \
			-d "Pruning parameter $PRUNE failed!" \
			-H "Title: Pruning parameter $PRUNE (Training: $TRAIN_BOOL)" \
			-H "Tags: warning" \
			"$NTFY_LINK";
			exit $EXIT_CODE;
		fi
	done
done | tqdm --total="$NUM_ITERATIONS" --desc="Ablation Pruning"

curl \
	-d "Running evaluation pipeline for $RUN_NAME. Results will be in $EVALUATION_DIRECTORY." \
	-H "Title: $RUN_NAME Evaluation" \
	-H "Tags: test_tube" \
	"$NTFY_LINK";

python3.14 -m washLTH -f="$EVALUATION_DIRECTORY" evaluate --dataset="$DATASET" --baseline="$MODEL" --max-evaluations=15 \
	--mask-filter="magnitude" --sql-filter="WHERE \"name\" REGEXP '$RUN_NAME \((Unt|T)rained\) \(\d+\.?\d*%\)'" --ablation --baseline-as-target;
curl -d "Finished evaluating with exit code $?" "$NTFY_LINK";
