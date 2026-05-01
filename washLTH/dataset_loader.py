from datasets import DatasetDict, load_dataset as base_load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import TokenizersBackend, BatchEncoding
from typing import overload

from .defaults import DEFAULT_SEED
from .types import DatasetType, DatasetDictType, LiveDatasetType, StreamingDatasetType, DatasetInfo
from .general import get_number_of_processors
from .hf_tools import HFObject


__all__ = ["load_dataset", "split_dataset", "DatasetType", "DatasetDictType"]


def format_dailymail(row: LazyBatch, tokenizer: TokenizersBackend) -> BatchEncoding:
	tokens = tokenizer(row["article"], row["highlights"], truncation=True)
	# print(tokens)
	return tokens


def format_mrpc(row: LazyBatch, tokenizer: TokenizersBackend) -> BatchEncoding:
	return tokenizer(row["sentence1"], row["sentence2"], truncation=True)


def format_arxivcap(row: LazyBatch, tokenizer: TokenizersBackend) -> BatchEncoding:
	eos_token = tokenizer.special_tokens_map.get("eos_token", "<EOS>")
	captions = [caption["caption"] for image in row["caption_images"] for caption in image]

	parsed = [
		eos_token.join(map(lambda pair: pair["sub_caption"], caption["cil_pairs"]))
		for image in row["caption_images"]
		for caption in image
	]

	tokens = tokenizer(captions, parsed, truncation=True, max_length=tokenizer.model_max_length)
	return tokens


def split_dataset(dataset: LiveDatasetType, seed: int = DEFAULT_SEED) -> DatasetDict:
	if isinstance(dataset, DatasetDict):
		dataset = dataset["train"]

	train_temp = dataset.train_test_split(test_size=0.3, seed=seed)
	temp_split = train_temp["test"].train_test_split(test_size=0.5, seed=seed)

	dataset: DatasetDict = DatasetDict({
		"train": train_temp["train"],
		"validation": temp_split["train"],
		"test": temp_split["test"],
	})

	return dataset


@overload
def load_dataset(dataset: str | HFObject, tokenizer, *args, streaming: bool = False, **kwargs) -> LiveDatasetType:
	...


@overload
def load_dataset(dataset: str | HFObject, tokenizer, *args, streaming: bool = True, **kwargs) -> StreamingDatasetType:
	...


@overload
def load_dataset(dataset: str | HFObject, tokenizer, *args, return_info: bool = False, **kwargs) -> LiveDatasetType:
	...


@overload
def load_dataset(dataset: str | HFObject, tokenizer, *args, return_info: bool = True, **kwargs) -> DatasetInfo:
	...


def load_dataset(dataset: str | HFObject, tokenizer, *args, seed=DEFAULT_SEED, streaming: bool = False, return_info: bool = False,
				 **kwargs):
	if isinstance(dataset, HFObject):
		kwargs.setdefault("revision", dataset.revision)
		dataset = dataset.name

	info = None

	if not streaming:
		kwargs.setdefault("num_proc", get_number_of_processors())

	if dataset.endswith(".parquet"):
		dataset = base_load_dataset("parquet", data_files=dataset, *args, streaming=streaming, **kwargs)
	else:
		match dataset:
			case "abisee/cnn_dailymail":
				dataset = base_load_dataset(dataset, "3.0.0", *args, streaming=streaming, **kwargs)
				if tokenizer is not None:
					dataset = dataset.map(format_dailymail, batched=True, batch_size=8, fn_kwargs=dict(tokenizer=tokenizer),
										  remove_columns=["id"])

				if return_info:
					info = DatasetInfo(dataset, format_dailymail)
			case "glue" | "mrpc":
				dataset = base_load_dataset("glue", "mrpc", *args, streaming=streaming, **kwargs)
				if tokenizer is not None:
					dataset = dataset.map(format_mrpc, batched=True, batch_size=8, fn_kwargs=dict(tokenizer=tokenizer),
										  remove_columns=["idx"])

				if return_info:
					info = DatasetInfo(dataset, format_mrpc)
			case "lwashington3/ArxivCap" | "lwashington3/ArxivCap-Minimal" | _:
				dataset = base_load_dataset(dataset, *args, streaming=streaming, **kwargs)
				column_names = next(iter(dataset.column_names.values()))
				# Need batched to be True to allow for each row (paper) to return the multiple captions it may contain
				if tokenizer is not None:
					dataset = dataset.map(format_arxivcap, batched=True, batch_size=1, fn_kwargs=dict(tokenizer=tokenizer),
										  remove_columns=column_names)
				if return_info:
					info = DatasetInfo(dataset, format_arxivcap)

	if isinstance(dataset, LiveDatasetType):
		dataset = split_dataset(dataset, seed=seed)
		if return_info:
			info.dataset = dataset

	if return_info:
		return info

	return dataset
