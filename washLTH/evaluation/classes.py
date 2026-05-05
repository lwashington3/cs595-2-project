from dataclasses import dataclass
from time import perf_counter_ns

import numpy as np
from transformers import TextGenerationPipeline, BatchEncoding, TextIteratorStreamer, GenerationConfig
from warnings import catch_warnings, filterwarnings

import torch
import torch.nn.functional as F


__all__ = ["LTHPipeline", "LTHStreamer", "ExperimentItem"]


class LTHStreamer(TextIteratorStreamer):
	"""Measures the Time to First Token (TTFT) of a model when passed to model.generate."""
	def __init__(self, *args, **kwargs):
		kwargs.setdefault("skip_prompt", True)
		kwargs.setdefault("skip_special_tokens", True)
		super().__init__(*args, **kwargs)
		self._ttft = None
		self._start_time_ns = None
		self._first_token_received = False

	@property
	def ttft_ns(self):
		"""The time to first token in nanoseconds."""
		return self._ttft

	@property
	def ttft(self):
		"""The time to first token in seconds."""
		if self._ttft is None:
			return None
		return self._ttft / 1e9

	def put(self, value):
		# Called once per generated token
		if not self._first_token_received:
			if self._start_time_ns is not None:
				self._ttft = perf_counter_ns() - self._start_time_ns
			self._first_token_received = True

		return super().put(value)

	def start(self):
		self._start_time_ns = perf_counter_ns()

	def __repr__(self):
		value = f"{self.ttft_ns:,} ns" if self.ttft_ns is not None else "None"
		return f"Time to First Token: {value}"


class LTHPipeline(TextGenerationPipeline):
	_default_generation_config = GenerationConfig(
		max_length=1000000000000,
		max_new_tokens=1
	)

	def preprocess(self, *args, **kwargs):
		process_start = perf_counter_ns()
		processed: BatchEncoding = super().preprocess(*args, **kwargs)
		processed.data["process_start"] = process_start
		return processed

	def _forward(self, inputs, **kwargs):
		streamer = LTHStreamer(self.tokenizer)
		streamer.start()

		generated_sequence = super()._forward(inputs, streamer=streamer, **kwargs)

		input_ids = inputs["input_ids"]
		target_ids = input_ids.clone()

		with catch_warnings():
			filterwarnings("ignore", category=UserWarning)
			with torch.no_grad():
				outputs = self.model(input_ids, labels=target_ids)

		# region Test
		logits = outputs.logits

		shift_logits = logits[:, :-1, :].contiguous()
		shift_labels = input_ids[:, 1:].contiguous()

		losses = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="none")
		losses = losses.view(input_ids.size(0), -1)

		# tokens = [self.tokenizer.decode(t) for t in input_ids[0]]
		# print(f"Last losses: {tuple(i.item() for i in losses[0, -50:])}")
		# endregion

		# generated_sequence["loss"] = outputs.loss
		generated_sequence["loss"] = losses.mean().item()
		generated_sequence["ttft"] = streamer.ttft
		generated_sequence["process_start"] = inputs.get("process_start")

		return generated_sequence

	def postprocess(self, outputs, **kwargs):
		results = super().postprocess(outputs, **kwargs)
		end_time = perf_counter_ns()
		if (start_time := outputs["process_start"]) is not None:
			run_time = (end_time - start_time) / 1e9
		else:
			run_time = torch.nan

		perplexity = np.exp(outputs["loss"])

		def assign_evaluations(dct: dict):
			dct["perplexity"] = perplexity
			dct["ttft"] = outputs["ttft"]
			dct["run_time"] = run_time
			output_tokens = outputs["generated_sequence"].shape[-1]
			dct["num_output_tokens"] = output_tokens
			dct["time_per_output_token"] = run_time / output_tokens
			dct["decoded_tokens"] = [self.tokenizer.decode(t) for t in outputs["input_ids"][0]]
		# Easier to see the (small) fixed length info before the resulting text
		# dct["generated_text"] = dct["generated_text"]

		if isinstance(results, list):
			for i in results:
				assign_evaluations(i)
		else:
			assign_evaluations(results)

		return results


@dataclass
class ExperimentItem:
	name: str
	base_model: str
	base_tensors: Optional[Path]
	mask: Optional[str | Path]
	tokenizer_directory: Path

	def __post_init__(self):
		if self.base_tensors is None:
			self.base_tensors = self.base_model

