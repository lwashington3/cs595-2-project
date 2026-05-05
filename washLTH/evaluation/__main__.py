from .classes import LTHPipeline, ExperimentItem
from .metrics import rogue_lcs
from .write_files import write_ablation_results, write_results
from ..tensors import apply_mask_to_safetensors, create_model_from_safetensors

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from pathlib import Path
from torch import OutOfMemoryError
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Callable, Iterable, Optional, Generator

import pandas as pd
import sqlite3

__all__ = ["evaluation_pipeline"]


def create_items_for_experiment(item: ExperimentItem) -> tuple[str, LTHPipeline]:
	tokenizer = AutoTokenizer.from_pretrained(item.tokenizer_directory)

	if isinstance(item.base_tensors, Path):
		masked_tensors = apply_mask_to_safetensors(item.base_tensors, item.mask)
		model = create_model_from_safetensors(item.base_tensors, safe_tensors=masked_tensors, strict=False)
	else:
		if item.mask is None:
			model = AutoModelForCausalLM.from_pretrained(item.base_model, device="cpu")
		else:
			model = create_model_from_safetensors(item.base_model, mask=item.mask, strict=False)

	model = model.cpu()
	model.generation_config.max_new_tokens = 1
	model.generation_config.max_length = None
	pipeline = LTHPipeline(model=model, tokenizer=tokenizer, max_new_tokens=model.generation_config.max_new_tokens, device="cpu")
	pipeline.generation_config.max_length = model.generation_config.max_length
	return item.name, pipeline


def test_models(prompt, experiments: tuple[tuple[str, LTHPipeline] | ExperimentItem, ...],
				device=None, max_new_tokens: int = None) -> pd.DataFrame:
	df = pd.DataFrame(index=[experiment[0] for experiment in experiments],
					  columns=["Perplexity", "ROGUE-L F1 Score", "TTFT (s)", "Run Time (s)", "Output Tokens",
							   "Time per Output Token (s)", "Generated Text"])

	pbar: tqdm[tuple[int, tuple[str, LTHPipeline] | ExperimentItem]] = tqdm(enumerate(experiments), desc="Running models through evaluation pipeline for row.", position=1)
	baseline_response = None
	decoded_tokens = [None] * len(experiments)

	for i, item in pbar:
		if isinstance(item, ExperimentItem):
			pbar.write(f"Run: {i=}, {item=}")
			name, pipeline = create_items_for_experiment(item)
		else:
			name, pipeline = item

		pipeline.model.to(device)
		results = pipeline(prompt)[0]
		pipeline.model.cpu()

		# TODO: Assuming the first one is the baseline
		df.loc[name] = [results["perplexity"], None, results["ttft"], results["run_time"], results["num_output_tokens"],
						results["time_per_output_token"], results["generated_text"]]
		decoded_tokens[i] = results["decoded_tokens"]
		if i == 0:
			baseline_response = results["decoded_tokens"]

	if baseline_response is not None:
		df["ROGUE-L F1 Score"] = [rogue_lcs(tokens, baseline_response) for tokens in decoded_tokens]
	else:
		df.drop(columns=["ROGUE-L F1 Score"], inplace=True)

	return df


def evaluation_pipeline(dataset: str | Dataset | DatasetDict, output_directory: Path, connection: sqlite3.Connection,
                        logger, ablation: bool = False, max_evaluations: int = None,
                        sql_filter: str = None, baseline: Optional[int | str] = None, max_new_tokens: int = None,
                        mask_selector: Callable[[Path], Iterable[Path]] = None):
	from ..dataset_loader import load_dataset
	output_directory.mkdir(parents=True, exist_ok=True)

	if isinstance(dataset, str):
		dataset = load_dataset(dataset, None, return_info=False)

	if isinstance(dataset, DatasetDict):
		if len(dataset.keys()) == 1:
			dataset: Dataset = dataset[tuple(dataset.keys())[0]]
		else:
			for key in dataset.keys():
				if "test" in key.lower():
					dataset: Dataset = dataset[key]
					break
			else:
				dataset: Dataset = dataset[tuple(dataset.keys())[0]]

	cursor = connection.cursor()

	if baseline is not None:
		if baseline.isdigit():
			baseline = int(baseline)
			baseline = cursor.execute("SELECT name FROM experiments WHERE ROWID = :row", dict(row=baseline)).fetchone()
		else:
			baseline_path = Path(baseline).resolve()

	device = Accelerator().device

	def iterator() -> Generator[ExperimentItem, None, None]:
		if baseline is not None:
			yield ExperimentItem(f"{baseline} (Baseline)", baseline, baseline_path if baseline_path.exists() else baseline, None, baseline)

		sql = "SELECT \"name\", base_model, pretrained_model, experiment_directory, tokenizer_directory FROM experiments"
		if sql_filter is not None:
			sql += f" WHERE {sql_filter.lstrip('WHERE ')}"

		logger.debug(f"SQL Query: {sql}")
		results = cursor.execute(sql)

		for name, base_model, pretrained_model, directory, tokenizer_directory in results.fetchall():
			directory = Path(directory).resolve()
			paths = mask_selector(directory) if mask_selector is not None else directory.rglob("**/mask.safetensors")

			base_tensors = Path(pretrained_model).resolve() if pretrained_model != "None" else None

			for mask_path in paths:
				yield ExperimentItem(name, base_model, base_tensors, mask_path.resolve(), tokenizer_directory)

	experiments = tuple(iterator())
	cursor.close()

	experiment_idx = 0
	try:
		items = [experiment for experiment in experiments]
		for experiment_idx, item in enumerate(experiments, start=experiment_idx):
			items[experiment_idx] = create_items_for_experiment(item)
			logger.debug(f"Created model {experiment_idx:,} of {len(experiments):,}: {item.name}.");
		logger.info(f"Created all models {len(experiments):,} and stored them in memory.")
	except (MemoryError, OutOfMemoryError) as e:
		logger.warning("Could not immediately load all items into memory. System will load each one individually, test it, then off load it.", exc_info=e)
		_slice = slice(experiment_idx - 1, experiment_idx)
		items[_slice] = experiments[_slice]

	if max_evaluations is not None:
		dataset = dataset.take(max_evaluations)

	latex_kwargs = dict(
		position="H",
		position_float="centering",
		hrules=True
	)
	latex_na_rep = "NaN"

	logger.info("Starting model evaluations.")
	for i, row in tqdm(enumerate(dataset, start=1), desc="Running Tests"):
		prompt = row["article"] # TODO: Have it get the correct column given the dataset
		df = test_models(prompt, items, device=device, max_new_tokens=max_new_tokens)

		if ablation:
			write_ablation_results(output_directory, df, i, prompt, latex_na_rep, latex_kwargs)
		else:
			write_results(output_directory, df, i, prompt, latex_na_rep, latex_kwargs)

		logger.debug(f"Finished testing row {i:,} of {len(dataset):,}.")

	logger.info("Model evaluations have finished.")
