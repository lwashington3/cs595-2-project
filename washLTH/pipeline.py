from pathlib import Path
from time import time_ns
from transformers import TextGenerationPipeline, AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any

import torch

__all__ = ["LTHPipeline", "running_pipeline"]


class LTHPipeline(TextGenerationPipeline):
	...


def test_pipeline(pipe: LTHPipeline) -> dict[str, Any]:
	prompt = "What can you tell me about Pompeii?"

	start = time_ns()
	response = pipe(prompt)
	run_time = time_ns() - start

	del pipe

	torch.cuda.empty_cache()
	free, total = torch.cuda.mem_get_info()
	print(f"Free: {free / 1e9:,.6f} GB, Total {total / 1e9:,.6f} GB.")

	return dict(
		response=response[0]["generated_text"],
		time=f"{run_time:_}"
	)


def running_pipeline(experiment_directory: Path):
	tokenizer = AutoTokenizer.from_pretrained(experiment_directory / "tokenizer")
	results_directory = experiment_directory / "results"
	results_directory.mkdir(parents=True, exist_ok=True)

	results = dict()

	for subdirectory in ("magnitude",):
		model = AutoModelForCausalLM.from_pretrained(experiment_directory / subdirectory)

		pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, pipeline_class=LTHPipeline)
		results[subdirectory] = test_pipeline(pipe)

		del model, pipe

	pipe = pipeline(model="meta-llama/Llama-3.1-8B-Instruct", tokenizer=tokenizer, pipeline_class=LTHPipeline)
	results["baseline"] = test_pipeline(pipe)

	with open(results_directory / f"{subdirectory}_results.json", "w") as f:
		from json import dump
		dump(results, f, indent="\t")
