import pandas as pd

from re import compile as re_compile
from pathlib import Path

from .plots import ablation_plot


__all__ = ["write_results", "write_ablation_results"]


def write_results(parent_directory: Path, df: pd.DataFrame, idx: int, prompt: str, latex_na_rep="NaN", latex_kwargs=None):
	latex_kwargs = latex_kwargs or dict()
	output_directory = parent_directory / f"{idx}"
	df.to_csv(output_directory / f"results.csv")

	styler: pd.io.formats.style.Styler = df.style
	styler = styler.highlight_min(subset=["Perplexity", "TTFT (s)", "Run Time (s)", "Time per Output Token (s)"],
	                              props="bfseries:;")
	styler = styler.highlight_max(subset=["ROGUE-L F1 Score"], props="bfseries:;")

	# TODO: Don't show Unnamed: 0 for the index name
	styler = styler.highlight_min(props="color:{red}", subset=(df.index[1:], df.columns[:-1]), axis=0)

	styler = styler.format(formatter="{:,.4f}".format, subset=["Perplexity"], na_rep=latex_na_rep)
	styler = styler.format(formatter="{:,.6f}".format, subset=["TTFT (s)", "Run Time (s)", "Time per Output Token (s)"],
	                       na_rep=latex_na_rep)
	styler = styler.format(formatter="{:,.0f}".format, subset=["Output Tokens"], na_rep=latex_na_rep)
	styler = styler.format(escape="latex", subset=["Generated Text"])
	styler = styler.format_index(escape="latex")
	styler.to_latex(output_directory / f"results_with_captions.tex", caption=f"Row {idx:,}: Caption: ``{prompt}''.",
	                label=f"tab:evaluation-pipeline-{idx}", **latex_kwargs)

	styler.hide(subset=["Generated Text"], axis=1)
	styler.to_latex(output_directory / f"results.tex", caption=f"Row {idx:,}: Caption: ``{prompt}''.",
	                label=f"tab:evaluation-pipeline-{idx}", **latex_kwargs)


def write_ablation_results(parent_directory: Path, df: pd.DataFrame, idx: int, prompt: str, latex_na_rep="NaN", latex_kwargs=None):
	write_results(parent_directory, df, idx, prompt, latex_na_rep, latex_kwargs)
	output_directory = parent_directory / f"{idx}"

	regex = re_compile(r"Ablation \d+ \(((Unt|T)rained)\) \((\d+\.?\d*%)\)")
	indices = [regex.search(i).groups() for i in df.index[1:]]
	indices = [(None, None)] + [(i[0] == "Trained", i[2]) for i in indices]

	df["Trained"] = [i[0] for i in indices]
	df["Pruning Percentage"] = [i[1] for i in indices]

	for column, file_name in (
			("Perplexity", None),
			("TTFT (s)", "ttft"),
			("Run Time (s)", "run_time"),
			("Time per Output Token (s)", "output_token_time"),
	):
		ablation_plot(df, output_directory, column, file_base=file_name)

	if "ROGUE-L F1 Score" in df.columns:
		ablation_plot(df, output_directory, "ROGUE-L F1 Score", file_base="rouge-l"),
