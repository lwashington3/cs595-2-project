#!/bin/bash

MODEL=google/gemma-3-1b-it
DATASET=abisee/cnn_dailymail
RESULTS_DIRECTORY=$(realpath $(dirname "$BASH_SCRIPT"))/models/bash

mkdir -p "$RESULTS_DIRECTORY";
echo "Results will be placed in: $RESULTS_DIRECTORY"

for PRUNE in $(seq 0.2 0.05 0.95) ; do
	curl \
		-d "Started pruning." \
		-H "Title: Pruning parameter $PRUNE" \
		-H "Tags: microscope" \
		"$NTFY_LINK"
	python3.14 -m washLTH -f="$RESULTS_DIRECTORY/$PRUNE" train --prune="$PRUNE" --model="$MODEL" --dataset="$DATASET" --name="Magnitude Prune ($(echo "$PRUNE * 100" | bc | sed -E 's/\.0{1,}$//')%)"
	EXIT_CODE=$?
	if [[ "$EXIT_CODE" -eq 0 ]]; then
	   curl \
		-d "Pruning parameter $PRUNE finished successfully!" \
		-H "Title: Pruning parameter $PRUNE" \
		-H "Tags: tada" \
		"$NTFY_LINK"
	else
	   curl \
		-d "Pruning parameter $PRUNE failed!" \
		-H "Title: Pruning parameter $PRUNE" \
		-H "Tags: warning" \
		"$NTFY_LINK";
		exit $EXIT_CODE;
	fi
done
