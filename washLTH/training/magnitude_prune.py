from .tools import correct_patterns, RegexType, StringChecker
from ..tensors import get_pytorch_device, SafeTensorReader
from ..types import SafeTensor

from datasets import Dataset
from pathlib import Path
from tqdm import tqdm
from typing import Iterable

import logging
import re
import torch


__all__ = ["magnitude_prune"]


def magnitude_prune(safe_tensors_file, pruning_parameter: float, logger: logging.Logger, train: Dataset = None,
					val: Dataset = None, output_file: Path = None, /,
					allow_layer_pattern: RegexType | StringChecker | Iterable[RegexType | StringChecker] = None,
					ignore_layer_pattern: RegexType | StringChecker | Iterable[RegexType | StringChecker] = None,
					**kwargs) -> SafeTensor:
	"""

	:param pathlib.Path safe_tensors_file: The safe tensor files.
	:param float pruning_parameter: The percentage of parameters to be masked out. Must be between 0 (exclusive) and 1 (exclusive).
	:param pathlib.Path output_file: If given, the masks will be written to a safetensors file.:
	:param str | re.Pattern allow_layer_pattern: If given, the regex will check if the name of a layer within the model matches it. If it does not, the layer will not have a mask created for it.
	:param str | re.Pattern ignore_layer_pattern: If given, the regex will check if the name of a layer within the model matches it. If it does, the layer will not have a mask created for it.
	:return:
	"""
	from functools import reduce
	from operator import mul

	if pruning_parameter <= 0 or pruning_parameter >= 1:
		raise ValueError(f"Pruning parameter must be in (0, 1), not {pruning_parameter:,}.")

	allow_layer_pattern: list[StringChecker] = correct_patterns(allow_layer_pattern)
	ignore_layer_pattern: list[StringChecker] = correct_patterns(ignore_layer_pattern)

	device: Device = kwargs.get("device", get_pytorch_device())

	# TODO: For iterative pruning to work, set up a while loop here and decrease the pruning parameter to decrease the number of usable parameters
	masks: SafeTensor = dict()

	with SafeTensorReader(safe_tensors_file, device=device) as reader:
		pbar = tqdm(reader, total=len(reader), desc="Generating Magnitude Masks")
		for layer in pbar:
			if ignore_layer_pattern and any(map(lambda ignore: ignore(layer), ignore_layer_pattern)):
				# If any of these are true, ignore this layer and continue the loop
				continue

			if allow_layer_pattern and not any(map(lambda allow: allow(layer), allow_layer_pattern)):
				# If all of these aren't true, then ignore this layer can continue the loop
				continue

			tensor = reader[layer]
			mask = torch.ones_like(tensor, dtype=torch.int8)

			layer_shape = tensor.shape
			num_parameters = reduce(mul, layer_shape)
			num_prune = num_parameters * pruning_parameter

			flattened = torch.flatten(tensor)
			idx = torch.argsort(torch.abs(flattened), descending=True).reshape(layer_shape)
			del flattened

			mask[idx > num_prune] = False
			masks[layer] = mask
			if device != "cpu":
				tensor.to("cpu")

	# TODO: Should test the results against the dataset if iteratively pruning

	if output_file is not None:
		from safetensors.torch import save_file

		logger.debug(f"Writing magnitude mask to {output_file}.")
		save_file(masks, output_file, metadata=kwargs.get("metadata"))

	return masks
