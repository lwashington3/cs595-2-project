from re import Pattern
from torch import Tensor, Device
from typing import Any, Callable


__all__ = ["Device", "MetricEvaluator", "RegexType", "SafeTensor", "StringChecker"]


MetricEvaluator = Callable[["transformers.trainer_utils.EvalPrediction"], dict[str, Any]]

RegexType = str | Pattern

SafeTensor = dict[str, Tensor]

StringChecker = Callable[[str], bool]

# DatasetMeta = tuple["datasets.dataset_dict.DatasetDict", DatasetInfo]
