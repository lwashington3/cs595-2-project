from washLTH import *
from washLTH.pipeline import check_perplexity

from pathlib import Path
from time import perf_counter_ns


def print_memory(info: str = ""):
	import torch

	free, total = torch.cuda.mem_get_info()
	print(f"{info}Free: {free / 1e9:,.6f} GB, Total {total / 1e9:,.6f} GB.")


def data():
	from washLTH.dataset_loader import generate_regular_dataset, weave_datasets

	data_parent = Path("/datasets/washcap").resolve()

	dataset = generate_regular_dataset(data_parent)
	weave_datasets(data_parent)


def main():
	base_model = "meta-llama/Llama-3.1-8B-Instruct"
	# safe_tensors_file = Path("/hf/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659/model.safetensors")
	experiment_directory = Path("/lth/models/testing")
	safe_tensors_file = experiment_directory / "model.safetensors"
	pruning_parameter = 0.3
	# tokenizer = experiment_directory / "tokenizer"

	mask_file = experiment_directory / "magnitude" / "mask.safetensors"
	device = get_pytorch_device()

	start = perf_counter_ns()
	# Magnitude pruning test
	# tensors = magnitude_prune(safe_tensors_file, pruning_parameter, mask_file, device=device)
	# print_memory("After pruning ")
	#
	# # Loading masked tensor files test
	# tensors = apply_mask_to_safetensors(safe_tensors_file, mask_file, device=device)
	# print_memory("After creating masked tensors ")
	#
	# # Load tensors into a model
	# model_start = perf_counter_ns()
	# model = create_model_from_safetensors(base_model, tensors, strict=True)
	# print_memory("After creating model")
	# print(f"Time to load model: {(perf_counter_ns() - model_start) / 1e9:,}s.")
	#
	# print(f"{model.device=}")

	# Load a sentence into the model to see if I can get results and maybe evaluate TTFT

	# config = AutoConfig.from_pretrained(base_model)
	generation_start = perf_counter_ns()

	evaluation_pipeline(experiment_directory)

	# model_inputs = tokenizer([""], return_tensors="pt").to(model.device)
	# generate_ids = model.generate(**model_inputs)
	# response = tokenizer.batch_decode(generate_ids)[0]
	# print(f"{response=}")
	end = perf_counter_ns()
	print(f"Total run time: {(end - start) / 1e9:,}s.\nTotal generation time: {(end - generation_start) / 1e9:,}s.")

	return tensors


if __name__ == "__main__":
	# tensors = main()
	# data()
	check_perplexity()
