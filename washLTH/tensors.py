from .types import Device, SafeTensor

from accelerate import Accelerator
from pathlib import Path
from safetensors import safe_open
from transformers import AutoModelForCausalLM
from typing import Optional
from tqdm import tqdm

import logging
import torch

__all__ = ["SafeTensor", "SafeTensorReader", "Device", "get_pytorch_device", "combine_safetensors", "apply_mask_to_safetensors", "create_model_from_safetensors"]


def get_pytorch_device(as_string=True) -> str:
	device = Accelerator().device
	if as_string:
		return str(device)
	return device
	# return "cuda" if torch.cuda.is_available() else "cpu"
	# return "cpu"


def combine_safetensors(output_file: Path = None, directory: Optional[Path] = None, tensor_files: list[str | Path] | tuple[str | Path, ...] = None,
                        metadata: dict[str, str] = None, device: Device = None) -> SafeTensor:
	from safetensors.torch import load_file, save_file

	if tensor_files is None and directory is None:
		raise ValueError("Either the directory holding the safe tensors of a list of the safetensors needs to be passed.")

	if tensor_files is None:
		tensor_files = tuple(directory.glob("*.safetensors"))

	# Might need to sort the layers, because the keys weren't in order
	tensors: SafeTensor = dict()
	device = device or get_pytorch_device()
	num_files = f"{len(tensor_files):,}"

	logging.debug(f"Beginning to load ({num_files}) safetensor files.")
	for i, file in enumerate(sorted(tensor_files)):
		tensor_slice = load_file(file, device=device)
		tensors.update(tensor_slice)
		logging.debug(f"Loaded safetensor file {i+1:,} of {num_files}: {file}")

	if output_file is not None:
		save_file(tensors, output_file, metadata=metadata)

	return tensors


class SafeTensorReader:
	def __init__(self, safetensor: SafeTensor | Path, device: Device = None):
		self.safetensor = safetensor
		self.device = device
		self.from_file = isinstance(self.safetensor, Path)

	def __getitem__(self, key: str) -> torch.Tensor:
		if self.from_file:
			return self.file.get_tensor(key)

		return self.safetensor[key]

	def __len__(self) -> int:
		if self.from_file:
			return len(self.file.offset_keys())

		return len(self.safetensor)

	def __enter__(self):
		if self.from_file:
			self.file = safe_open(self.safetensor, framework="pt", device=self.device)
			self.file.__enter__()

		return self

	def __exit__(self, *args, **kwargs):
		if self.from_file:
			self.file.__exit__(*args, **kwargs)

	def __iter__(self):
		if self.from_file:
			for key in self.file.offset_keys():
				yield key
		else:
			for key in self.safetensor.keys():
				yield key

	def __contains__(self, key: str) -> bool:
		if self.from_file:
			return key in self.file.offset_keys()
		return key in self.safetensor.keys()


def apply_mask_to_safetensors(safe_tensors: SafeTensor | Path, masks: SafeTensor | Path, device: Device = None) -> SafeTensor:
	"""
	Reads in a safetensors file with the associated mask file.
	:param pathlib.Path safe_tensors: The path to the safetensors file.
	:param pathlib.Path masks: The file where the masks for the safetensors are, or the safe tensors already read into memory.
	:param str device: The PyTorch device to load the tensors to.
	:return:
	:rtype:
	"""
	new_tensors: SafeTensor = dict()
	device = device or get_pytorch_device()

	with SafeTensorReader(safe_tensors, device=device) as tensors, SafeTensorReader(masks, device=device) as masks:
		for key in tqdm(tensors, total=len(tensors), desc="Loading Masked Layers"):
			safe_layer = tensors[key]

			if key not in masks:
				new_tensors[key] = safe_layer
				continue

			mask = masks[key]

			new_tensors[key] = safe_layer * mask
			# if device != "cpu" and tensors.from_file:
			# 	safe_layer.to("cpu")

	return new_tensors


def write_masked_model(tensors: SafeTensor, output_file: Path, metadata: dict[str, str] = None):
	from safetensors.torch import save_file

	save_file(tensors, output_file, metadata=metadata)


def create_model_from_safetensors(base_model: str | Path, safe_tensors: str | SafeTensor, strict=True, **kwargs) -> AutoModelForCausalLM:
	from transformers import AutoConfig

	config = AutoConfig.from_pretrained(base_model, **kwargs)

	if isinstance(safe_tensors, str):
		from safetensors.torch import load_file
		safe_tensors = load_file(safe_tensors)

	# try:
	model = AutoModelForCausalLM.from_config(config)
	model.load_state_dict(safe_tensors, strict=strict, assign=True) # TODO: Catch the error here so I can catch it and try again with the meta wrapper
	# with torch.device("meta"):
	# 	model = AutoModelForCausalLM.from_config(config)
	#
	# model.load_state_dict(safe_tensors, strict=strict, assign=True)

	return model
