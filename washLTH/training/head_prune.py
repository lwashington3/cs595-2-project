from ..types import SafeTensor
from .magnitude_prune import magnitude_prune

import re

__all__ = ["head_prune"]


def head_prune(*args, **kwargs) -> SafeTensor:
	"""Does magnitude pruning of only the head layer(s) by passing an allow to :func:`~washLTH.training.magnitude_prune`"""
	kwargs["allow_layer_pattern"] = re.compile(r"^.*head.*$", flags=re.IGNORECASE)
	# kwargs["ignore_layer_pattern"] = lambda string: not re.compile(r"^.*head.*$", flags=re.IGNORECASE).match(string)

	return magnitude_prune(*args, **kwargs)
