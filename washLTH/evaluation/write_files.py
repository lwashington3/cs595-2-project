import numpy as np
import pandas as pd

from matplotlib import ticker
from pathlib import Path
from re import compile as re_compile
from typing import Any

from .plots import plot_all_ablations


__all__ = ["write_results", "write_ablation_results"]


def write_results(output_directory: Path, df: pd.DataFrame, idx: int, prompt: str, latex_na_rep: str = "-",
				  latex_kwargs: dict[str, Any] = None, show_generated_text: bool = False):
	latex_kwargs = latex_kwargs or dict()

	df.to_csv(output_directory / f"results.csv")
	latex_kwargs.setdefault("column_format", None)
	column_format = latex_kwargs.pop("column_format", None)
	ablation_column_format = latex_kwargs.pop("ablation_column_format", None)

	df["ROGUE-L F1 Score"] = df["ROGUE-L F1 Score"] * 100
	columns = list(df.columns)
	position = columns.index("ROGUE-L F1 Score")
	columns[position] = "ROGUE-L F1 Score (%)"

	for i, column in enumerate(df.columns):
		if "(s)" not in column:
			continue
		df[column] = df[column] * 1000
		columns[i] = column.replace("(s)", "(ms)")

	df.columns = columns

	styler: pd.io.formats.style.Styler = df.style
	styler = styler.highlight_min(subset=["Perplexity", "TTFT (ms)", "Run Time (ms)", "Time per Output Token (ms)"], props="bfseries:;")
	styler = styler.highlight_max(subset=["ROGUE-L F1 Score (%)"], props="bfseries:;")

	styler = styler.highlight_min(props="color:{red}", subset=pd.IndexSlice[df.index[1:], ["Perplexity", "TTFT (ms)", "Run Time (ms)", "Time per Output Token (ms)"]], axis=0)
	styler = styler.highlight_max(props="color:{red}", subset=pd.IndexSlice[df.index[1:], ["ROGUE-L F1 Score (%)"]], axis=0)

	styler = styler.format(formatter="{:,.4f}".format, subset=["Perplexity"], na_rep=latex_na_rep)
	styler = styler.format(formatter="{:,.4f}\\%".format, subset=["ROGUE-L F1 Score (%)"], na_rep=latex_na_rep)
	styler = styler.format(formatter="{:,.3f}".format, subset=["TTFT (ms)", "Run Time (ms)", "Time per Output Token (ms)"], na_rep=latex_na_rep)
	styler = styler.format(formatter="{:,.0f}".format, subset=["Output Tokens"], na_rep=latex_na_rep)

	styler = styler.format_index(escape="latex", axis=0).format_index(escape="latex", axis=1)
	if show_generated_text:
		styler = styler.format(escape="latex", subset=["Generated Text"])
		styler.to_latex(output_directory / f"results_with_captions.tex", caption=f"Row {idx:,}\\ifshowcaption{{}}: Caption: ``{prompt}''.\\else.",
						label=f"tab:evaluation-pipeline-{idx}", column_format=column_format, **latex_kwargs)

	styler_hidden = styler.hide(subset=["Generated Text"], axis=1)

	styler_hidden.to_latex(output_directory / f"results.tex", caption=f"Row {idx:,}\\ifshowcaption{{}}: Caption: ``{prompt}''.\\else.\\fi",
	                label=f"tab:evaluation-pipeline-{idx}", column_format=ablation_column_format,
					**latex_kwargs)

	train_index = df.index.get_level_values("Trained")
	styler_hidden.hide(subset=df.index[train_index == False], axis=0)\
		.to_latex(output_directory / f"trained_results.tex", caption=f"Row {idx:,} Results (Pre-trained before Pruning). \\ifshowcaption{{}}Caption: ``{prompt}''.\\fi",
	                label=f"tab:evaluation-pipeline-{idx}-trained", column_format=ablation_column_format,
					**latex_kwargs)

	styler_hidden.hide(subset=df.index[train_index == True], axis=0)\
		.to_latex(output_directory / f"untrained_results.tex", caption=f"Row {idx:,} Results (No pre-training before pruning). \\ifshowcaption{{}}Caption: ``{prompt}''.\\fi",
				  label=f"tab:evaluation-pipeline-{idx}-untrained", column_format=ablation_column_format,
					**latex_kwargs)


def write_ablation_results(results_directory: Path, df: pd.DataFrame, idx: int, prompt: str, latex_na_rep: str = "-", latex_kwargs: dict[str, Any] = None):
	regex = re_compile(r"Ablation \d+ \(((Unt|T)rained)\) \((\d+\.?\d*%)\)")
	remove_extra_zeroes = re_compile(r"(\.[1-9]+)0*(%)?$")

	indices = [regex.search(i).groups() for i in df.index[1:]]
	indices = [(np.nan, "0%")] + [(i[0] == "Trained", remove_extra_zeroes.sub(r"\g<1>\2", i[2])) for i in indices]

	indices = pd.MultiIndex.from_tuples(indices, names=["Trained", "Pruning Percentage"])
	df.set_index(indices, inplace=True)

	write_results(results_directory, df, idx, prompt, latex_na_rep=latex_na_rep, latex_kwargs=latex_kwargs)

	df["Trained"] = [i[0] for i in indices]
	df["Pruning Percentage"] = [i[1] for i in indices]

	plot_kwargs = dict(
		major_formatter=lambda x, _: f"{100 * x:.0f}",
		major_locator=ticker.MultipleLocator(0.05),
		minor_locator=ticker.MultipleLocator(0.025)
	)
	plot_all_ablations(df, results_directory, plot_kwargs=plot_kwargs)
