from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict
from datasets.formatting.formatting import LazyBatch
from re import Pattern
from torch import Tensor
from torch.types import Device
from transformers import TokenizersBackend, BatchEncoding, EvalPrediction
from typing import Any, Callable, NamedTuple


__all__ = ["DatasetType", "DatasetDictType", "LiveDatasetType", "StreamingDatasetType", "FormatterType", "Device",
		   "MetricEvaluator", "RegexType", "SafeTensor", "StringChecker", "DatasetInfo"]

DatasetType = Dataset | IterableDataset

DatasetDictType = DatasetDict | IterableDatasetDict

LiveDatasetType = Dataset | DatasetDict

StreamingDatasetType = IterableDataset | IterableDatasetDict

FormatterType = Callable[[LazyBatch, TokenizersBackend], BatchEncoding]


class DatasetInfo(NamedTuple):
	dataset: DatasetType | DatasetDictType
	formatter: FormatterType


MetricEvaluator = Callable[[EvalPrediction], dict[str, Any]]

RegexType = str | Pattern

SafeTensor = dict[str, Tensor]

StringChecker = Callable[[str], bool]

# DatasetMeta = tuple["datasets.dataset_dict.DatasetDict", DatasetInfo]
