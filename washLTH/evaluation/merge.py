import pandas as pd

from enum import Enum
from functools import reduce
from matplotlib import ticker
from operator import add
from pathlib import Path
from typing import Callable

from .plots import plot_all_ablations
from .write_files import write_results

__all__ = ["merge_results"]


class MergeMethods(Enum):
	Average = (lambda df, num_dfs: df / num_dfs)


def merge_results(results: tuple[pd.DataFrame | Path], output_directory: Path, idx: int, prompt: str,
				  merge_method: MergeMethods | Callable[[pd.DataFrame, int], pd.DataFrame] = MergeMethods.Average) -> pd.DataFrame:
	num_dfs = len(results)
	dfs = [None] * num_dfs
	for i, result in enumerate(results):
		if isinstance(result, Path):
			if not result.is_file():
				raise FileNotFoundError(f"File `{result.resolve()}` does not exist")
			result = pd.read_csv(result, index_col=[0, 1])

		if "Generated Text" in result.columns:
			result.drop("Generated Text", axis=1, inplace=True)
		dfs[i] = result

	merged_df: pd.DataFrame = reduce(add, dfs)
	merged_df = merge_method(merged_df, num_dfs)
	# merged_df.columns = [f"Average {column}" for column in merged_df.columns]

	write_results(output_directory, merged_df, idx, prompt, show_generated_text=False)

	plot_kwargs = dict(
		major_formatter=lambda x, _: f"{100 * x:.0f}",
		major_locator=ticker.MultipleLocator(0.05),
		minor_locator=ticker.MultipleLocator(0.025)
	)
	plot_all_ablations(merged_df, output_directory, plot_kwargs=plot_kwargs)

	return merged_df
