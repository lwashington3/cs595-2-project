from ..types import RegexType, StringChecker
from typing import Iterable, Optional


import re


__all__ = ["correct_patterns", "RegexType", "StringChecker"]


def correct_patterns(pattern: Optional[RegexType | StringChecker | Iterable[StringChecker | RegexType]]) -> list[StringChecker]:
	"""
	Forces compliance of regex strings, regex objects, or any objects with a similar structure to check if a string matches some internal pattern.
	:return: A list of callables that will check if a given string matches some internal pattern.
	:rtype: list[StringChecker]
	"""
	if pattern is None:
		return list()

	elif not hasattr(pattern, "__iter__") or isinstance(pattern, str):
		return correct_patterns([pattern])

	patterns = []

	for p in pattern:
		if isinstance(p, str):
			patterns.append(re.compile(p).match)
		elif isinstance(p, re.Pattern):
			patterns.append(p.match)
		elif callable(p):
			patterns.append(p)
		else:
			raise ValueError(f"Pattern must be a string or a regex, not {type(p).__name__}.")

	return patterns
