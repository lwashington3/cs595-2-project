from difflib import SequenceMatcher
from typing import Sequence


__all__ = ["rogue_lcs"]


def rogue_lcs(predicted: str | Sequence[str], target: str | Sequence[str]) -> float:
	if isinstance(predicted, str):
		predicted = predicted.split()

	if isinstance(target, str):
		target = target.split()

	num_predicted = len(predicted)
	num_target = len(target)

	lcs_len = SequenceMatcher(None, target, predicted).find_longest_match().size

	precision = lcs_len / num_predicted
	recall = lcs_len / num_target

	if lcs_len == 0:
		return 0

	return (2 * precision * recall) / (precision + recall)
