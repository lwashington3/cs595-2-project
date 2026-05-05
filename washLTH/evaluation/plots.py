import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, rcParams
from pathlib import Path


__all__ = ["perplexity_plot", "ablation_plot"]


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


def ablation_plot(df: pd.DataFrame, output_folder: Path, column: str, row: int = None, add_bar_labels: bool = False,
				  file_base: str = None, **kwargs):
	figure, ax = plt.subplots(nrows=1, ncols=1)

	trained = df[df["Trained"] == True]
	untrained = df[df["Trained"] == False]
	baseline = df[pd.isna(df["Trained"])]

	trained_color = kwargs.get("trained_color", "C0")
	untrained_color = kwargs.get("untrained_color", "C2")
	baseline_color = kwargs.get("baseline_color", "C3")

	names = ["Baseline"] + trained["Pruning Percentage"].to_list()

	x = np.arange(len(names))
	ax.set_xticks(x)
	ax.set_xticklabels(labels=names)
	ax.set_xlabel("Pruning Percentage")

	width = 0.25

	for multiplier, (label, values, color) in enumerate((
		("Trained", trained[column], trained_color),
		("Untrained", untrained[column], untrained_color),
	)):
		offset = width * multiplier
		rects = ax.bar(x[1:] + offset, values, width, label=label, color=color)
		if add_bar_labels:
			ax.bar_label(rects, padding=3, color=color)

	rects = ax.bar(x[0], baseline[column], 2*width, label="Baseline", color=baseline_color)
	if add_bar_labels:
		ax.bar_label(rects, padding=3, color=baseline_color)

	ax.set_ylabel(column)

	if row is None:
		ax.set_title(column)
	else:
		ax.set_title(f"{column} for Row {row:,}")

	ax.legend()
	figure.tight_layout()
	file_base = file_base or column.replace(" ", "_")
	for _format, extension in DEFAULT_FIGURE_FORMATS.items():
		figure.savefig(output_folder / f"{file_base}.{extension}", format=_format)


def time_plot():
	ax2 = ax.twinx()
	ax2.plot(x, df["TTFT (s)"] * 1000, label="TTFT (ms)", c="C2", marker="o")
	ax2.set_ylabel("TTFT (ms)", color="C2")

	figure.suptitle(f"Perplexity vs TTFT (ms) for Row {row:,}")
	# figure.legend()
	figure.tight_layout()
	figure.savefig(output_file_base, transparent=True)


if __name__ == "__main__":
	dark_mode()
	directory = Path("/lth/evaluation/bash_pruning")
	for i in range(4):
		perplexity_plot(pd.read_csv(directory / f"{i}.csv").set_index("Unnamed: 0"), directory / f"{i}.png", i)
