__all__ = ["ttft_vs_perplexity"]

import re

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt, rcParams
from pathlib import Path


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


def ttft_vs_perplexity(df: pd.DataFrame, output_file_base: Path, row: int):
	figure, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 8))

	regex = re.compile(r"Magnitude Prune \(0.(\d)%\)")

	df = df.drop(labels=["run_2026-04-19T22:00:26.926557"])

	names = ["Baseline"] + [regex.sub(r"\g<1>0%", index) for index in df.index[1:]]
	x = np.arange(len(df.index))
	ax.set_xticks(x)
	ax.set_xticklabels(labels=names)
	ax.set_xlabel("Pruning Percentage")

	ax.plot(x, df["Perplexity"] / 1e6, label="Perplexity", c="C1", marker="o")
	ax.set_ylabel("Perplexity (millions)", color="C1")

	ax2 = ax.twinx()
	ax2.plot(x, df["TTFT (s)"] * 1000, label="TTFT (ms)", c="C2", marker="o")
	ax2.set_ylabel("TTFT (ms)", color="C2")

	# for x_pos, x_label, perplexity, ttft in zip(x, df.index, df["Perplexity"], df["TTFT (s)"]):
	# 	ax.scatter(x_pos, perplexity, label=x_label)
	# 	ax2.scatter(x_pos, ttft, label=x_label)

	figure.suptitle(f"Perplexity vs TTFT (ms) for Row {row:,}")
	# figure.legend()
	figure.tight_layout()
	figure.savefig(output_file_base, transparent=True)


if __name__ == "__main__":
	dark_mode()
	directory = Path("/lth/evaluation/bash_pruning")
	for i in range(4):
		ttft_vs_perplexity(pd.read_csv(directory / f"{i}.csv").set_index("Unnamed: 0"), directory / f"{i}.png", i)
