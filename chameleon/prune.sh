#!/bin/bash

MODEL=google/gemma-3-1b-it
DATASET=abisee/cnn_dailymail
RESULTS_DIRECTORY=$(realpath $(dirname "$BASH_SCRIPT"))/models/bash

echo "Results will be placed in: $RESULTS_DIRECTORY"
mkdir -p "$RESULTS_DIRECTORY";

for PRUNE in $(seq 0.5 0.1 0.9) ; do
	curl \
		-d "Started pruning." \
		-H "Title: Pruning parameter $PRUNE" \
		-H "Tags: microscope" \
		"$NTFY_LINK"
	python3 -m washLTH -f="$RESULTS_DIRECTORY/$PRUNE" train --prune="$PRUNE" --model="$MODEL" --dataset="$DATASET" --name="Magnitude Prune ($PRUNE%)"
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
