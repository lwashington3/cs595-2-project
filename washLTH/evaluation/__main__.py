from .classes import LTHPipeline
from ..tensors import create_model_from_safetensors

from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Iterable

import pandas as pd
import sqlite3

__all__ = ["evaluation_pipeline"]


def check_perplexity():
	from datasets import load_dataset

	device = Accelerator().device
	# model_id = "openai-community/gpt2-large"
	# model_id = "meta-llama/Llama-3.1-8B-Instruct"
	model_id = "google/gemma-3-1b-it"
	model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
	tokenizer = AutoTokenizer.from_pretrained(model_id)

	test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
	prompt = "\n\n".join(test["text"])

	pipe = LTHPipeline(model=model, tokenizer=tokenizer)
	# output = pipe(["How are you today?", "What is your favorite language?"])
	for row in prompt:
		output = pipe(row)
		print(output[0])
		# for i in output[0]:
		# 	print(i)


def test_models(prompt, experiments: Iterable[tuple[str, str | Path, str | Path]], device=None) -> pd.DataFrame:
	df = pd.DataFrame(index=[name for name, *_ in experiments], columns=["Perplexity", "TTFT (s)", "Run Time (s)", "Generated Text"])

	old_tokenizer_directory = None
	old_tokenizer = None

	for i, (name, base_model, tensors_file, tokenizer_directory) in tqdm(enumerate(experiments), desc="Running models through evaluation pipeline for row.", position=1):
		if tokenizer_directory == old_tokenizer_directory:
			tokenizer = old_tokenizer
		else:
			tokenizer = AutoTokenizer.from_pretrained(tokenizer_directory)

		# TODO: It looks like the pipeline is adding extra text when it doesn't need to, which may explain why the perplexity is so high
		if base_model == tensors_file:
			model = AutoModelForCausalLM.from_pretrained(base_model).to(device)
		else:
			model = create_model_from_safetensors(base_model, tensors_file, strict=False)
		pipe = LTHPipeline(model=model, tokenizer=tokenizer)
		results = pipe(prompt)[0]

		df.loc[name] = [results["perplexity"], results["ttft"], results["run_time"], results["generated_text"]]

		old_tokenizer_directory = tokenizer_directory
		old_tokenizer = tokenizer_directory

	return df


def evaluation_pipeline(dataset: str | Dataset | DatasetDict, output_directory: Path, connection: sqlite3.Connection,
						logger, base_model: str = None, max_evaluations: int = None, sql_filter: str = None):
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

	device = Accelerator().device

	cursor = connection.cursor()
	if base_model is not None:
		def iterator(): # TODO: Merge base model into the where
			yield f"{base_model} (Baseline)", base_model, base_model, base_model

			sql = "SELECT \"name\", base_model, safetensors_file, tokenizer_directory FROM experiments"
			if sql_filter is not None:
				sql += f"WHERE {sql.lstrip('WHERE ')}"
			else:
				sql += f"WHERE base_model = '{base_model}'"

			yield from cursor.execute(sql).fetchall()
	else:
		def iterator():
			yield from cursor.execute(f"SELECT \"name\", base_model, safetensors_file, tokenizer_directory FROM experiments").fetchall()

	experiments = tuple(iterator())
	cursor.close()

	if max_evaluations is not None:
		dataset = dataset.take(max_evaluations)

	logger.info("Starting model evaluations.")
	for i, row in tqdm(enumerate(dataset), desc="Running Tests"):
		prompt = row["article"] # TODO: Have it get the correct column given the dataset
		df = test_models(prompt, experiments, device=device)
		df.to_csv(output_directory / f"{i}.csv")

		styler: pd.io.formats.style.Styler = df.style
		styler = styler.highlight_min(subset=["Perplexity", "TTFT (s)", "Run Time (s)"], props="bfseries:;")
		styler.format(escape="latex").to_latex(output_directory / f"{i}.tex",
		                caption=f"Row {i:,}: Caption: ``{prompt}''.",
		                label=f"tab:evaluation-pipeline-{i}")
		logger.debug(f"Finished testing row {i:,} of {len(dataset):,}.")

	logger.info("Model evaluations have finished.")


if __name__ == "__main__":
	check_perplexity()
