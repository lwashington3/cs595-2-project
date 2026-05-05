from ..hf_tools import *
from ..dataset_loader import load_dataset
from ..defaults import DEFAULT_SEED
from ..tensors import apply_mask_to_safetensors, create_model_from_safetensors, combine_safetensors, get_pytorch_device
from ..types import *

from datasets import Dataset, DatasetDict
from huggingface_hub import repo_type_and_id_from_hf_id
from huggingface_hub.serialization._base import MAX_SHARD_SIZE
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

import logging


__all__ = ["do_magnitude_pruning", "training_pipeline"]


MAX_TOKENIZER_LENGTH = 131_072 # Tokenizer length for Llama 3.1


def do_magnitude_pruning(model: HFObject, experiment_directory: Path, safetensor_file: Path, pruning_parameter: float,
						 device: Device, logger: logging.Logger, train: Dataset = None, val: Dataset = None,
						 prune_head_only: bool = True, save_masked_model: bool = False, **kwargs) -> tuple[SafeTensor, AutoModelForCausalLM]:
	if not prune_head_only: # Only did it this way so the documentation for magnitude_prune would show up for pruner
		from .magnitude_prune import magnitude_prune as pruner
		log_value = "magnitude"
	else:
		from .head_prune import head_prune as pruner
		log_value = "head"

	directory = experiment_directory / log_value
	directory.mkdir(parents=True, exist_ok=True)
	mask_file = directory / "mask.safetensors"

	mask = pruner(safetensor_file, pruning_parameter, logger, train, val, mask_file, **kwargs)
	logger.info(f"Generated the masks for {log_value} pruning and wrote the mask to the experiment directory.")

	tensors = apply_mask_to_safetensors(safetensor_file, mask, device=device)
	logger.info(f"Applied the {log_value} masks to the safetensors.")

	model: AutoModelForCausalLM = create_model_from_safetensors(model.name, safe_tensors=tensors, strict=False, revision=model.revision)
	logger.info(f"Created the {log_value} model given the {log_value} mask.")

	if save_masked_model:
		model.save_pretrained(directory, max_shard_size=kwargs.get("max_shard_size", MAX_SHARD_SIZE))
		logger.info(f"Saved the {log_value} model using `save_pretrained()`.")

	return mask, model


def training_pipeline(model: HFObject, experiment_directory: Path, pruning_parameter: float, dataset: Optional[HFObject | DatasetDict] = None, /,
					  logger: logging.Logger = None, **kwargs):
	"""

	:param str model: The baseline model that is being used in the process
	:param str dataset: The dataset that is being used to pretrain the model
	:param pathlib.Path experiment_directory: The parent directory where resulting models and other artifacts are stored
	:param float pruning_parameter: The percentage of neurons to prune (0, 1).
	:param logging.Logger logger:
	:param str training_key: The key for the training dataset within the dataset (if a DatasetDict is given). Default is train.
	:param str validation_key: The key for the validation dataset within the dataset (if a DatasetDict is given). Default is val.
	"""

	from .pretrain import pretrain_model

	logger = logger or logging.getLogger("lottery_ticket_hypothesis")
	logger.info(f"Started training pipeline with model={model} and dataset={dataset}")
	kwargs.setdefault("device", get_pytorch_device())
	kwargs.setdefault("push_to_hub", False)
	kwargs.setdefault("seed", DEFAULT_SEED)

	device: Device = kwargs.get("device")
	# seed: int = kwargs["seed"]
	training_key = kwargs.get("training_key", "train")
	validation_key = kwargs.get("validation_key", "validation")

	experiment_directory.mkdir(parents=True, exist_ok=True)
	snapshot_path: Path = pull_model(model)

	# Combines the safe tensors shards into a single file
	# safetensor_file = experiment_directory / "model.safetensors"
	safetensor_file = snapshot_path / "combined_model.safetensors"
	combine_safetensors(output_file=safetensor_file, directory=snapshot_path)
	logger.info("Combined safetensors into one file.")

	# Gets the tokenizer associated with the model and saves it for future use
	tokenizer = AutoTokenizer.from_pretrained(snapshot_path, use_safetensors=True)

	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
		tokenizer.pad_token_id = tokenizer.eos_token_id

	# if tokenizer.model_max_length is None:
	# 	tokenizer.model_max_length = MAX_TOKENIZER_LENGTH
	# elif tokenizer.model_max_length > MAX_TOKENIZER_LENGTH:
	# 	tokenizer.model_max_length = MAX_TOKENIZER_LENGTH

	tokenizer.save_pretrained(experiment_directory / "tokenizer")
	logger.info("Wrote tokenizer to experiment directory.")

	# Loads the dataset
	if isinstance(dataset, HFObject):
		streaming = kwargs.get("streaming", False)
		# dataset_name = dataset.name
		dataset = load_dataset(dataset.name, tokenizer, revision=dataset.revision, download_mode="reuse_dataset_if_exists",
							   streaming=streaming)
		# if not streaming:
		# 	dataset: DatasetDict = split_dataset(dataset, seed)

		logger.info("Loaded and preprocessed the dataset.")

	if dataset is None:
		train, val = None, None
		pretrained_safe_tensors = safetensor_file
	else:
		train: Dataset = dataset[training_key].take(6)
		val: Dataset = dataset[validation_key].take(4)
		dataset_name = train.info.dataset_name

		# Pretrains a version of the model on the given dataset
		# pretrained_directory = experiment_directory / f"pretrained_{dataset_name}" # TODO: Add username at some point if possible
		pretrained_directory = snapshot_path / f"pretrained_{dataset_name}"
		if pretrained_directory.exists():
			pretrained_safe_tensors = pretrained_directory / "model.safetensors"
		else:
			pretrained_directory.mkdir(parents=True, exist_ok=True)
			_, _, pretrained_safe_tensors, pretrained_model, _ = pretrain_model(snapshot_path, train, val, tokenizer,
																			pretrained_directory, **kwargs)

		# Creates a link to where it's base safetensors are, to reduce how often files are written
		(experiment_directory / "base_model.safetensors").symlink_to(pretrained_safe_tensors)
		logger.info(f"Pretrained the model and saved to {pretrained_directory}.")

	# Create a list of arguments required for magnitude pruning (so I'm not repeating code)
	kwargs.pop("device")

	# Prune the entire model given generalized magnitude pruning
	pruning_args = (model, experiment_directory, pretrained_safe_tensors, pruning_parameter, device, logger, train, val)
	do_magnitude_pruning(*pruning_args, prune_head_only=False, **kwargs)

	do_magnitude_pruning(*pruning_args, prune_head_only=True, **kwargs)

	# structure_model = structure_prune(pretrained_model, dataset, experiment_directory)

	return safetensor_file, pretrained_safe_tensors if dataset is not None else None
