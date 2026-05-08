# CS 595 Project: Using the Lottery Ticket Hypothesis to Decrease the Size of LLMs
Len Washington III

## Using the Docker Container
To create a Docker container with the necessary dependencies, use
```shell
docker compose run -it project bash
```
to enter into the terminal.

## Ablation Script
To run the full ablation script from start to finish, run [ablation](scripts/ablation).
This will pretrain the model, prune it with values 2.5% to 97.5%, then evaluation each of them.

## Training Pipeline
To run the training pipeline (that takes in a model and dataset and prunes  the weights), run within the container
```shell
python3 -m washLTH -f={The folder where the results will be stored} --model={The model from HuggingFace Hub being used} --dataset={The dataset from HuggingFace being used}
```

For example:
```shell
python3 -m washLTH -f=./models/testing train --model=meta-llama/Llama-3.1-8B-Instruct --dataset=lwashington3/ArxivCap
```

Running
```shell
python3 -m washLTH --help
```
can be used to understand more about the options

## Evaluation Pipeline
To evaluate the pruned models, run
```shell
python3 -m washLTH -f={The folder where the results will be stored} evaluate --model=google/gemma-3-1b-it --dataset=abisee/cnn_dailymail --max-evaluations=15 --sql-filter="WHERE \"name\" REGEXP '$RUN_NAME \((Unt|T)rained\) \(\d+\.?\d*%\)'" --ablation 
```

## Merged Evaluation Results
To merge previous evaluation results (assuming the same model/dataset combo, though it's not required), use the [merge](scripts/merge) script to combine them all.

## Pruning Parameter Mix-Up
I did not realize until 1778195738 that how I had the pruning parameter was inverted.
It was describing what percentage of parameters should remain, instead of what percentage of parameters to mask.
This change should be reflected in the [report](out/washington_lth.pdf), but any figures or tables from beforehand will show percentage remaining instead of percentage pruned.