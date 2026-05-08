import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, rcParams
from operator import add, sub
from pathlib import Path
from re import compile as re_compile

__all__ = ["perplexity_plot", "ablation_plot", "plot_all_ablations"]


DEFAULT_FIGURE_FORMATS = {"png": "png", "svg": "svg", "pgf": "pgf"}
rcParams.update({
	"figure.figsize": (12, 8),
	"savefig.transparent": True,
	"legend.framealpha": 0
})


def dark_mode():
	edge_color = "white"
	params = {
		"text.color": edge_color,
		"axes.edgecolor": edge_color,
		"axes.labelcolor": edge_color,
		"figure.edgecolor": edge_color,
		"figure.facecolor": "#ffffff40",
		"xtick.color": edge_color,
		"ytick.color": edge_color,
	}
	rcParams.update(params)
	return params


def perplexity_plot(df: pd.DataFrame, output_folder: Path, row: int):
	figure, ax = plt.subplots(nrows=1, ncols=1)

	names = ["Baseline"] + df.columns[1:]
	x = np.arange(len(df.index))
	ax.set_xticks(x)
	ax.set_xticklabels(labels=names)
	ax.set_xlabel("Pruning Percentage")

	ax.plot(x, df["Perplexity"], label="Perplexity", c="C1", marker="o")
	ax.set_ylabel("Perplexity", color="C1")

	figure.suptitle(f"Perplexity for Row {row:,}")
	# figure.legend()
	figure.tight_layout()
	for _format, extension in DEFAULT_FIGURE_FORMATS.items():
		figure.savefig(output_folder / f"perplexity.{extension}", format=_format)
	plt.close(figure)


def ablation_plot(df: pd.DataFrame, output_folder: Path, column: str, row: int = None, add_bar_labels: bool = False,
				  file_base: str = None, show_baseline: bool = True, **kwargs):
	figure, ax = plt.subplots(nrows=1, ncols=1)

	is_trained = df.index.get_level_values("Trained")
	pruning_percentage = df.index.get_level_values("Pruning Percentage")

	trained = df[is_trained == True]
	untrained = df[is_trained == False]
	baseline = df[pd.isna(is_trained)]

	trained_color = kwargs.get("trained_color", "C2")
	untrained_color = kwargs.get("untrained_color", "C3")
	baseline_color = kwargs.get("baseline_color", "C0")

	names = pruning_percentage.unique()
	zero_check = re_compile(r"^0+\.?0*?")

	x = [float(name.replace("%", '')) / 100 for name in names if zero_check.match(name) is None]
	if show_baseline:
		x = [0.0] + x
	# elif not show_baseline:
	# 	x = x[1:]
	x = np.asarray(x)

	ax.set_xticks(x)

	if (formatter := kwargs.get("major_formatter", None)) is not None:
		ax.xaxis.set_major_formatter(formatter)

	if (formatter := kwargs.get("minor_formatter", None)) is not None:
		ax.xaxis.set_minor_formatter(formatter)

	if (locator := kwargs.get("major_locator", None)) is not None:
		ax.xaxis.set_major_locator(locator)

	if (locator := kwargs.get("minor_locator", None)) is not None:
		ax.xaxis.set_minor_locator(locator)

	ax.set_xlabel("Pruning Percentage")

	width = 0.01
	ablation_index = x[1:] if show_baseline else x

	for multiplier, (label, values, color, offset_func) in enumerate((
			("Trained", trained[column], trained_color, sub),
			("Untrained", untrained[column], untrained_color, add),
	)):
		# offset = width * multiplier
		offset = width / 2
		rects = ax.bar(offset_func(ablation_index, offset), values, width, label=label, color=color)
		if add_bar_labels:
			ax.bar_label(rects, padding=3, color=color)

	if show_baseline:
		rects = ax.bar(x[0], baseline[column], 2*width, label="Baseline", color=baseline_color)
		if add_bar_labels:
			ax.bar_label(rects, padding=3, color=baseline_color)

	ax.set_ylabel(column)

	if row is None:
		ax.set_title(column)
	else:
		ax.set_title(f"{column} for Row {row:,}")

	ax.set_xlim(x[0] - (2*width), x[-1] + (2*width))
	ax.legend()
	figure.tight_layout()

	if file_base is None:
		file_base = column.replace(" ", "_")
		if show_baseline:
			file_base += "_baseline"

	for _format, extension in DEFAULT_FIGURE_FORMATS.items():
		figure.savefig(output_folder / f"{file_base}.{extension}", format=_format)
	plt.close(figure)


def plot_all_ablations(df: pd.DataFrame, results_directory: Path, plot_kwargs=None):
	plot_kwargs = plot_kwargs or {}
	try:
		for column, file_name in (
				("Perplexity", "perplexity"),
				("TTFT (ms)", "ttft"),
				("Run Time (ms)", "run_time"),
				("Time per Output Token (ms)", "output_token_time"),
		):
			for show_baseline in (True, False):
				ablation_plot(df, results_directory, column, file_base=file_name + ("_baseline" if show_baseline else ""), show_baseline=show_baseline, **plot_kwargs)

		column = "ROGUE-L F1 Score (%)"
		if column in df.columns:
			ablation_plot(df, results_directory, column, file_base="rogue_l_baseline", show_baseline=True, **plot_kwargs)
			ablation_plot(df, results_directory, column, file_base="rogue_l", show_baseline=False, **plot_kwargs)
	except ValueError as e:
		raise ValueError(f"Error while plotting ablation: {column}, {show_baseline}: {e}") from e


if __name__ == "__main__":
	dark_mode()
	directory = Path("/lth/evaluation/bash_pruning")
	for i in range(4):
		perplexity_plot(pd.read_csv(directory / f"{i}.csv").set_index("Unnamed: 0"), directory / f"{i}.png", i)
