from pathlib import Path
from typing import Callable, Iterable

__all__ = ["SELECTOR_LOOKUP", "magnitude_only", "head_only", "structured_only"]


def magnitude_only(directory: Path) -> Iterable[Path]:
	mask = directory / "magnitude" / "mask.safetensors"
	if not mask.exists():
		raise FileNotFoundError(f"{directory} does not exist.")
	return (mask,)


def head_only(directory: Path) -> Iterable[Path]:
	mask = directory / "head" / "mask.safetensors"
	if not mask.exists():
		raise FileNotFoundError(f"{directory} does not exist.")
	return (mask,)


def structured_only(directory: Path) -> Iterable[Path]:
	raise NotImplementedError("Have not yet set up structural pruning in training, so I don't know what directory it'll save to.")
	# mask = directory / "structured" / "mask.safetensors"
	# if not mask.exists():
	# 	raise FileNotFoundError(f"{directory} does not exist.")
	# return (mask,)


SELECTOR_LOOKUP: dict[str, Callable[[Path], Iterable[Path]]] = {
	"magnitude": magnitude_only,
	"head": head_only,
	"structured": structured_only,
}
