from ..hf_tools import *
from ..dataset_loader import load_dataset
from ..tensors import apply_mask_to_safetensors, create_model_from_safetensors
from ..types import *

from datasets import DatasetDict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, EvalPrediction
from typing import Any

import logging
import torch


__all__ = ["compute_metrics", "pretrain_model", "magnitude_prune", "training_pipeline"]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
	metric = evaluate.load("glue", "mrpc")
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


def do_magnitude_pruning(model: HFObject, experiment_directory: Path, safetensor_file: Path, dataset: DatasetDict,
						 pruning_parameter: float, device: Device, prune_head_only: bool = True,
						 **kwargs) -> tuple[SafeTensor, AutoModelForCausalLM]:
	if not prune_head_only: # Only did it this way so the documentation for magnitude_prune would show up for pruner
		from .magnitude_prune import magnitude_prune as pruner
		log_value = "magnitude"
	else:
		from .head_prune import head_prune as pruner
		log_value = "head"

	directory = experiment_directory / log_value
	directory.mkdir(parents=True, exist_ok=True)
	mask_file = directory / "mask.safetensors"

	mask = pruner(safetensor_file, pruning_parameter, dataset, mask_file, **kwargs)
	logging.info(f"Generated the masks for {log_value} pruning and wrote the mask to the experiment directory.")

	tensors = apply_mask_to_safetensors(safetensor_file, mask, device=device)
	logging.info(f"Applied the {log_value} masks to the safetensors.")

	model: AutoModelForCausalLM = create_model_from_safetensors(model.name, tensors, revision=model.revision)
	logging.info(f"Created the {log_value} model given the {log_value} mask.")

	model.save_pretrained(directory, max_shard_size=kwargs.get("max_shard_size"))  # Use max_shard_size="5GB" to auto split it
	logging.info(f"Saved the {log_value} model using `save_pretrained()`.")
	return mask, model


def training_pipeline(model: HFObject, dataset: HFObject, experiment_directory: Path, pruning_parameter: float, /,
					  logger: logging.Logger = None, **kwargs):
	"""

	:param str model: The baseline model that is being used in the process
	:param str dataset: The dataset that is being used to pretrain the model
	:param pathlib.Path experiment_directory: The parent directory where resulting models and other artifacts are stored
	:param logging.Logger logger:
	"""

	from .pretrain import pretrain_model

	logger = logger or logging.getLogger("lottery_ticket_hypothesis")
	kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
	device: Device = kwargs["device"]
	kwargs.setdefault("push_to_hub", False)

	experiment_directory.mkdir(parents=True, exist_ok=True)
	snapshot_path: Path = pull_model(model)

	# Combines the safe tensors shards into a single file
	safetensor_file = experiment_directory / "model.safetensors"
	combine_safetensors(output_file=safetensor_file, directory=snapshot_path)
	logger.info("Combined safetensors into one file.")

	# Gets the tokenizer associated with the model and saves it for future use
	tokenizer = AutoTokenizer.from_pretrained(snapshot_path, use_safetensors=True)
	tokenizer.save_pretrained(experiment_directory / "tokenizer")
	logger.info("Wrote tokenizer to experiment directory.")

	# Loads the dataset
	dataset: DataSetDict = load_dataset(dataset.name, tokenizer, revision=dataset.revision, download_mode="reuse_dataset_if_exists")
	logger.info("Loaded and preprocessed the dataset.")

	# Pretrains a version of the model on the given dataset
	pretrained_directory = experiment_directory / "pretrained"
	pretrained_directory.mkdir(parents=True, exist_ok=True)
	pretrained_model = pretrain_model(snapshot_path, dataset, tokenizer, pretrained_directory, **kwargs)
	logger.info("Pretrained the model.")

	# Create a list of arguments required for magnitude pruning (so I'm not repeating code)
	pruning_args = (model, experiment_directory, safetensor_file, dataset, pruning_parameter, device)

	# Prune the entire model given generalized magnitude pruning
	magnitude_mask, magnitude_model = do_magnitude_pruning(*pruning_args, prune_head_only=False, **kwargs)

	head_mask, head_model = do_magnitude_pruning(*pruning_args, prune_head_only=True, **kwargs)

	# structure_model = structure_prune(pretrained_model, dataset, experiment_directory)


if __name__ == "__main__":
	main()
