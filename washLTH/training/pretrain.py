from ..tensors import get_pytorch_device
from ..types import MetricEvaluator

from datasets import Dataset
from pathlib import Path
from torch import OutOfMemoryError, cuda
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, DataCollatorForLanguageModeling, Trainer, EvalPrediction
from transformers.trainer_utils import TrainOutput
from typing import Any

import evaluate
import gc
import logging
import numpy as np
import sys


__all__ = ["compute_metrics", "pretrain_model", "pretrain_model_iteratively"]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
	metric = evaluate.load("glue", "mrpc")
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


def _train_rows_iteratively(logger: logging.Logger, model: AutoModelForCausalLM | Path, train: Dataset, val: Dataset, tokenizer: PreTrainedTokenizerBase,
                   output_directory: Path, lower_bound: int = None, upper_bound: int = None, training_kwargs: dict[str, Any] = None,
                   compute_metrics_func: MetricEvaluator = compute_metrics, **kwargs) -> tuple[Trainer, TrainOutput, Path, AutoModelForCausalLM, DataCollatorForLanguageModeling, int]:
	lower_bound = lower_bound or 0
	upper_bound = upper_bound or len(train)

	if lower_bound == upper_bound:
		return *pretrain_model(logger, model, train.take(lower_bound), val.take(lower_bound), tokenizer, output_directory,
		                       training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs), lower_bound
	elif lower_bound > upper_bound:
		lower_bound, upper_bound = upper_bound, lower_bound

	if lower_bound + 1 == upper_bound:
		try:
			values = pretrain_model(logger, model, train.take(upper_bound), val.take(upper_bound), tokenizer, output_directory,
									training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs)
			return *values, upper_bound
		except OutOfMemoryError:
			return _train_rows_iteratively(logger, model, train, val, tokenizer, output_directory, lower_bound=lower_bound, upper_bound=lower_bound,
										   training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs)

	middle = (lower_bound + upper_bound) // 2
	val_lines = min(middle, len(val))
	train_partition = train.take(middle)
	val_partition = val.take(val_lines)

	try:
		logger.debug(f"Successfully trained {middle:,} rows, will increase the number until convergence.")
		pretrain_model(logger, model, train_partition, val_partition, tokenizer, output_directory, training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs)
		return _train_rows_iteratively(logger, model, train, val, tokenizer, output_directory, lower_bound=middle, upper_bound=upper_bound, training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs)
	except OutOfMemoryError as e:
		if middle == 1:
			raise OutOfMemoryError("The device is too small to allow any training rows for this model.") from e
		logger.debug(f"{middle:,} training rows were too many.")
		return _train_rows_iteratively(logger, model, train, val, tokenizer, output_directory, lower_bound=lower_bound, upper_bound=middle, training_kwargs=training_kwargs, compute_metrics_func=compute_metrics_func, **kwargs)


def pretrain_model_iteratively(logger: logging.Logger, model: AutoModelForCausalLM | Path, train: Dataset, val: Dataset, tokenizer: PreTrainedTokenizerBase,
                   output_directory: Path, training_kwargs: dict[str, Any] = None, compute_metrics_func: MetricEvaluator = compute_metrics,
							   **kwargs) -> tuple[Trainer, TrainOutput, Path, AutoModelForCausalLM, DataCollatorForLanguageModeling, tuple[int, int]]:
	*values, lines = _train_rows_iteratively(logger, model, train, val, tokenizer, output_directory, training_kwargs=training_kwargs,
											 compute_metrics_func=compute_metrics_func, **kwargs)
	return *values, (lines, min(lines, len(val)))


def pretrain_model(logger: logging.Logger, model: AutoModelForCausalLM | Path, train: Dataset, val: Dataset, tokenizer: PreTrainedTokenizerBase,
                   output_directory: Path, training_kwargs: dict[str, Any] = None,
                   compute_metrics_func: MetricEvaluator = compute_metrics, **kwargs) -> tuple[Trainer, TrainOutput, Path, AutoModelForCausalLM, DataCollatorForLanguageModeling]:
	"""Pretrains a given model based on a given dataset"""
	from transformers import TrainingArguments, logging as tlogging

	device = kwargs.get("device", get_pytorch_device())
	logger.info("IN PRETRAIN_MODEL")
	if isinstance(model, Path):
		model = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True, device_map="auto")

	directory_exists = output_directory.exists()
	training_args = TrainingArguments(
		str(output_directory),
		use_cpu=device == "cpu",
		push_to_hub=kwargs.get("push_to_hub", False)
	)

	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

	training_kwargs = training_kwargs or dict()

	model.to(device)
	trainer = Trainer(
		model,
		training_args,
		train_dataset=train,
		eval_dataset=val,
		data_collator=data_collator,
		processing_class=tokenizer,
		compute_metrics=compute_metrics_func,
		# callbacks=[ProgressCallback()],
		**training_kwargs
	)

	trainer.model.to(device)

	verbosity = tlogging.get_verbosity()
	tlogging.enable_progress_bar()
	tlogging.set_verbosity_info()

	logger.info("ABOUT TO TRAIN!")
	try:
		training_output = trainer.train()
	except OutOfMemoryError as e:
		if not directory_exists: # In case the model doesn't train, the next run doesn't try to train with a file that doesn't exist
			output_directory.rmdir()

		message = e.args[0]
		logger.warning(f"The device ran out of memory while training: {message}", exc_info=None)

		trainer.model.cpu()
		del trainer.model
		del trainer

		e.__traceback__ = None # Having the exception keeps all of the tensors on the GPU, which would then crash every other iteration
		sys.exc_info() # Flushes the interpreter's internal exc state

		gc.collect()
		cuda.empty_cache()

		raise OutOfMemoryError(message)
	logger.info("Finished training.")

	trainer.model.save_pretrained(output_directory, safe_serialization=True)

	config = trainer.model.generation_config
	config.update(dict(max_new_tokens=20))
	config.save_pretrained(output_directory, safe_serialization=True)

	pretrained_safetensors_file = output_directory / "model.safetensors"
	if not pretrained_safetensors_file.exists():
		raise FileNotFoundError(f"{pretrained_safetensors_file} does not exist.")

	tlogging.set_verbosity(verbosity)

	return trainer, training_output, pretrained_safetensors_file, model, data_collator
