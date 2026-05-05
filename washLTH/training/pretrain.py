from ..tensors import get_pytorch_device
from ..types import MetricEvaluator

from datasets import Dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase, DataCollatorForLanguageModeling, Trainer, EvalPrediction
from transformers.trainer_utils import TrainOutput
from typing import Any

import evaluate
import logging
import numpy as np


__all__ = ["compute_metrics", "pretrain_model"]


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, Any]:
	metric = evaluate.load("glue", "mrpc")
	logits, labels = eval_pred
	predictions = np.argmax(logits, axis=-1)
	return metric.compute(predictions=predictions, references=labels)


def pretrain_model(model: AutoModelForCausalLM | Path, train: Dataset, val: Dataset, tokenizer: PreTrainedTokenizerBase,
                   output_directory: Path, training_kwargs: dict[str, Any] = None,
                   compute_metrics_func: MetricEvaluator = compute_metrics, **kwargs) -> tuple[Trainer, TrainOutput, Path, AutoModelForCausalLM, DataCollatorForLanguageModeling]:
	"""Pretrains a given model based on a given dataset"""
	from transformers import TrainingArguments, logging as tlogging

	device = kwargs.get("device", get_pytorch_device())
	logging.info("IN PRETRAIN_MODEL")
	if isinstance(model, Path):
		model = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True, device_map="auto")

	training_args = TrainingArguments(
		str(output_directory),
		use_cpu=device == "cpu", # or True,
		push_to_hub=kwargs.get("push_to_hub", False)
	)

	data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)

	training_kwargs = training_kwargs or dict()

	print(f"{device=}")
	model.to(device) # TODO: Might want to try "google/gemma-3-1b-it"
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

	logging.info("ABOUT TO TRAIN!")
	training_output = trainer.train()
	logging.info("Finished training.")

	trainer.model.save_pretrained(output_directory, safe_serialization=True)

	config = trainer.model.generation_config
	config.update(dict(max_new_tokens=20))
	config.save_pretrained(output_directory, safe_serialization=True)

	pretrained_safetensors_file = output_directory / "model.safetensors"
	if not pretrained_safetensors_file.exists():
		raise FileNotFoundError(f"{pretrained_safetensors_file} does not exist.")

	tlogging.set_verbosity(verbosity)

	return trainer, training_output, pretrained_safetensors_file, model, data_collator
