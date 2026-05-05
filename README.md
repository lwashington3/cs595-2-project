# CS 595 Project: Using the Lottery Ticket Hypothesis to Decrease the Size of LLMs
Len Washington III

## Using the Docker Container
To create a Docker container with the necessary dependencies, use
```shell
docker compose run -it project bash
```
to enter into the terminal.

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
python3 -m washLTH -f={The folder where the results will be stored} evaluate --model=google/gemma-3-1b-it --dataset=abisee/cnn_dailymail --max-evaluations=15
```