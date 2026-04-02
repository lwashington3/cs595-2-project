from datasets import DatasetDict, load_dataset as base_load_dataset
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorWithPadding

from .hf_tools import HFObject


__all__ = ["DatasetMeta", "load_dataset"]


# class DatasetInfo(NamedTuple):
# 	"""A class that holds information related to a dataset."""
# 	tokenizer: TokenizersBackend
# 	collator: DataCollator
# 	tokenize_func: Callable # TODO: Get the types being taken and returned
# 	compute_metrics: Callable[[EvalPrediction], dict[str, Any]]


DEFAULT_TOKENIZER = "bert-base-uncased"
DEFAULT_SEED = 20_484_223


def download_files(repo_id: str, revision, local_directory: Path):
	from huggingface_hub import snapshot_download

	snapshot_download(repo_id, revision=revision, repo_type="dataset", local_dir=local_directory,
	                  allow_patterns=[f"arXiv_src_0{i}*.parquet" for i in range(4)])
	cached_files = tuple(local_directory.glob("*/data/*.parquet"))
	return cached_files


def get_subcaptions(row):
	captions = row["caption_images"][0]
	pairs = captions["cil_pairs"]
	return any(map(lambda pair: bool(pair["sub_caption"]), pairs))


def download_dataset():
	dataset = base_load_dataset("MMInstruction/ArxivCap", split="train", cache_dir="/datasets")
	# TODO: After having downloaded the dataset to the hard drive, use IterableDataset to load it row by row for the filtering


def generate_regular_dataset(skip=0):
	from httpx import post
	from datasets import load_dataset as base_load_dataset, Dataset, IterableDataset
	from multiprocessing import cpu_count
	from tqdm import tqdm
	# from tqdm.contrib.discord import tqdm
	from typing import Generator

	NTFY_LINK = "https://ntfy.lenwashingtoniii.com/lottery-ticket-hypothesis"

	processors = cpu_count() - 2
	original_dataset: IterableDataset = base_load_dataset("MMInstruction/ArxivCap", split="train", streaming=True)
	original_dataset.remove_columns(["src", "title", "abstract", "meta"])
	features = original_dataset.features
	num_rows = original_dataset.info.splits.total_num_examples
	pbar = tqdm(original_dataset, total=num_rows, desc="Filtering rows")

	if skip:
		original_dataset = original_dataset.skip(skip)
		pbar.update(skip)

	def generator(bar: tqdm) -> Generator[dict]:
		for i, row in enumerate(bar):
			if i % 1000 == 0:
				post(NTFY_LINK, content=f"Passing index {i:,}.")
			if get_subcaptions(row):
				print(f"[{i:,}] ID: {row['arxiv_id']} can be used.")
				yield row

	try:
		dataset = Dataset.from_generator(generator, gen_kwargs=dict(bar=pbar), features=features)
		dataset.save_to_disk("/lth/datasets/washcap")
		commit_info = dataset.push_to_hub("lwashington3/ArxivCap", commit_message="Filtered out the rows with non-null subcaptions from MMInstruction/ArxivCap.")

		post(NTFY_LINK, content=f"Dataset pushed successfully to {commit_info.commit_url}.")
		return dataset
	except BaseException as e:
		from traceback import format_exc
		post(NTFY_LINK, content=f"Generating dataset did not work: `{e}`\n```{format_exc()}```.")
	finally:
		if "dataset" in locals():
			dataset.save_to_disk("/lth/datasets/washcap")


def generate_dataset(repo_id: str, message: str, revision: str = None, **kwargs):
	from os import getenv

	local_directory = Path(getenv("HF_HOME")).resolve() / "hub" / "datasets--MMInstruction--ArxivCap" / "snapshots"
	cached_files = []
	if revision is not None:
		local_directory = local_directory / revision
	else:
		directories = sorted(filter(lambda directory: directory.is_dir(), local_directory.glob("*")),
							 key=lambda directory: directory.stat().st_mtime, reverse=True)
		if len(directories) == 0:
			cached_files = download_files(repo_id, revision, local_directory)
		else:
			local_directory = directories[0]
	cached_files = cached_files or tuple(local_directory.glob("data/*.parquet"))

	if len(cached_files) == 0:
		cached_files = download_files(repo_id, revision, local_directory)
	cached_files = tuple(map(str, filter(lambda file: file.exists(), cached_files)))

	dataset = base_load_dataset("parquet", data_files=cached_files)

	train = dataset["train"].remove_columns(["src", "title", "abstract", "meta"])

	def get_subcaptions(row):
		captions = row["caption_images"][0]
		pairs = captions["cil_pairs"]
		return any(map(lambda pair: bool(pair["sub_caption"]), pairs))

	for i, row in enumerate(train):
		print(i, get_subcaptions(row))

	train_temp = train.train_test_split(test_size=0.3, seed=kwargs.get("split_seed"))
	temp_split = train_temp["test"].train_test_split(test_size=0.5, seed=kwargs.get("split_seed"))

	dataset: DatasetDict = DatasetDict({
		"train": train_temp["train"],
		"validation": temp_split["train"],
		"test": temp_split["test"],
	})

	# TODO: ValueError: No columns in the dataset match the model's forward method signature: (input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, logits_to_keep, kwargs, label, labels, label_ids). The following columns have been ignored: [title, src, abstract, meta, arxiv_id, caption_images]. Please check the dataset and model. You may need to set `remove_unused_columns=False` in `TrainingArguments`.

	dataset.push_to_hub(repo_id, commit_message=message)
	return dataset


# def load_arxivcap(dataset: str, tokenizer, local_directory: Path = None, **kwargs) -> DatasetDict:
# 	from os import getenv
#
# 	local_directory = local_directory or Path(getenv("HF_HOME")).resolve() / "hub" / "datasets--MMInstruction--ArxivCap" / "snapshots"
# 	cached_files = tuple(local_directory.glob("*/data/*.parquet"))
#
# 	if len(cached_files) == 0:
# 		from huggingface_hub import snapshot_download
#
# 		snapshot_download(dataset, repo_type="dataset", local_dir=local_directory,
# 						  allow_patterns=[f"arXiv_src_0{i}*.parquet" for i in range(4)])
# 		cached_files = tuple(local_directory.glob("*/data/*.parquet"))
#
# 	cached_files = tuple(map(str, filter(lambda file: file.exists(), cached_files)))
# 	dataset = base_load_dataset("parquet", data_files=cached_files)
#
# 	train_temp = dataset["train"].train_test_split(test_size=0.3, seed=kwargs.get("split_seed"))
# 	temp_split = train_temp["test"].train_test_split(test_size=0.5, seed=kwargs.get("split_seed"))
#
# 	dataset: DatasetDict = DatasetDict({
# 		"train": train_temp["train"],
# 		"validation": temp_split["train"],
# 		"test": temp_split["test"],
# 	})
#
# 	return dataset


def load_dataset(dataset: str | HFObject, tokenizer, *args, **kwargs) -> DatasetDict:
	if isinstance(dataset, HFObject):
		kwargs.setdefault("revision", dataset.revision)
		dataset = dataset.name

	kwargs.setdefault("split_seed", 20_484_223)

	match dataset:
		case "MMInstruction/ArxivCap":
			return load_arxivcap(dataset, tokenizer, *args, **kwargs)
		case "lwashington3/exsclaim":
			raise NotImplementedError("I have not yet implemented my dataset.")
		case "lwashington3/ArxivCap" | _:
			return base_load_dataset(dataset, *args, **kwargs)
