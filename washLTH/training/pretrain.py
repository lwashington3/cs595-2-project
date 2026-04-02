from ..tensors import get_pytorch_device
from ..types import MetricEvaluator

from datasets import DatasetDict
from pathlib import Path
from transformers import AutoModelForCausalLM, DataCollatorWithPadding, PreTrainedTokenizerBase
from typing import Any

import logging


__all__ = ["pretrain_model"]


def pretrain_model(model: AutoModelForCausalLM | Path, dataset: DatasetDict, tokenizer: PreTrainedTokenizerBase,
                   output_directory: Path, training_kwargs: dict[str, Any] = None,
                   compute_metrics_func: MetricEvaluator = None, **kwargs):
	"""Pretrains a given model based on a given dataset"""
	from transformers import TrainingArguments, Trainer

	logging.info("IN PRETRAIN_MODEL")
	if isinstance(model, Path):
		model = AutoModelForCausalLM.from_pretrained(model, use_safetensors=True)

	training_args = TrainingArguments(
		str(output_directory),
		push_to_hub=kwargs.get("push_to_hub", False)
	)

	data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

	training_kwargs = training_kwargs or dict()

	trainer = Trainer(
		model,
		training_args,
		train_dataset=dataset["train"],
		eval_dataset=dataset["validation"],
		data_collator=data_collator,
		processing_class=tokenizer,
		compute_metrics=compute_metrics_func or compute_metrics,
		**training_kwargs
	)

	trainer.model.to(kwargs.get("device", get_pytorch_device()))
	logging.info("ABOUT TO TRAIN!")
	trainer.train()

	return trainer, model, tokenizer, data_collator
