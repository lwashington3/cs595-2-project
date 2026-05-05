from .defaults import DEFAULT_SEED
from .hf_tools import HFObject, login
from .training import training_pipeline
from .evaluation import evaluation_pipeline, SELECTOR_LOOKUP
from .database import get_connection, safe_cursor

from pathlib import Path
from psutil import virtual_memory
from torch.cuda import is_available

import logging


old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
	record = old_factory(*args, **kwargs)

	mem = virtual_memory()
	conversion = 2**30 # (2^10)^3
	_format = kwargs.get("byte_format", ",.6f")

	record.total_memory = f"{mem.total / conversion:{_format}} GB"
	record.available_memory = f"{mem.available / conversion:{_format}} GB"
	record.free_memory = f"{mem.free / conversion:{_format}} GB"
	record.used_memory = f"{mem.used / conversion:{_format}} GB"

	if is_available():
		from torch.cuda import mem_get_info

		free, total = mem_get_info()
		used = total - free

		record.free_gpu_memory = f"{free / conversion:{_format}} GB"
		record.used_gpu_memory = f"{used / conversion:{_format}} GB"
		record.total_gpu_memory = f"{total / conversion:{_format}} GB"
	return record


def create_logger(directory: Path, log_file: str) -> logging.Logger:
	from sys import stdout
	from torch.cuda import is_available
	import logging.config

	directory.mkdir(exist_ok=True, parents=True)

	print_format = "%(asctime)s\t%(pathname)s:%(lineno)3d | RAM: %(used_memory)s used. %(available_memory)s of %(total_memory)s available.{}: %(message)s".format(" | GPU Usage: (%(used_gpu_memory)s of %(total_gpu_memory)s used)" if is_available() else "")
	file_format = "%(levelname)5s %(asctime)s.%(msecs)03d | PID: %(process)s | Thread: %(thread)d | %(name)s | Function: %(funcName)s() in %(pathname)s on line %(lineno)3d | RAM: %(used_memory)s used. %(available_memory)s of %(total_memory)s available.{} | %(message)s".format(" | GPU Usage: (%(used_gpu_memory)s of %(total_gpu_memory)s used)" if is_available() else "")

	logging.config.dictConfig({
		"version": 1,
		"disable_existing_loggers": True,
		"formatters": {
			"print": {
				"format": print_format,
				"datefmt": "%m/%d/%Y %H:%M:%S"
			},
			"file": {
				"format": file_format,
				"datefmt": "%m/%d/%Y %H:%M:%S"
			}
		},
		"handlers": {
			"console": {
				"class": "logging.StreamHandler",
				"formatter": "print",
				"level": "DEBUG",
				"stream": stdout
			},
			"file": {
				"class": "logging.FileHandler",
				"formatter": "file",
				"level": "DEBUG",
				"filename": str(directory / log_file),
				"mode": "w"
			}
		},
		"loggers": {
			__name__: {
				"level": "DEBUG",
				"handlers": ["console", "file"],
				"propagate": False,
			}
		}
	})

	logging.setLogRecordFactory(record_factory)
	return logging.getLogger(__name__)


def main(args=None):
	from argparse import ArgumentParser
	from datetime import datetime as dt

	default_model = "meta-llama/Llama-3.1-8B-Instruct"
	default_dataset = "lwashington3/ArxivCap-Minimal"
	# default_dataset = "lwashington3/exsclaim-caption-distributor"

	parser = ArgumentParser(prog="washLTH")

	subparsers = parser.add_subparsers(dest="command", required=True)
	train_parser = subparsers.add_parser("train", help="Run the training pipeline.")
	evaluate_parser = subparsers.add_parser("evaluate", help="Run a pruned model.")
	dataset_parser = subparsers.add_parser("dataset", help="Pre-process the dataset and upload it to HuggingFace Hub.")

	parser.add_argument("-t", "--token", default=None,
						help="Your HuggingFace token. If not given the system will pull it from the \"HF_TOKEN\" environment variable.")
	parser.add_argument("-f", "--folder", type=Path, default=Path("models") / f"run_{dt.now().isoformat()}",
						help="The folder where models and datasets related to this run are downloaded and stored.")
	parser.add_argument("-l", "--log", type=str, default="LTH.log", help="The name of the log directory.")
	parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="The seed of the pseudo random number generator.")
	parser.add_argument("-db", "--sqlite3-file", type=Path, default=Path(__file__).parent.parent / "LTH.db",
						help="The SQLite database file to hold information about experiments between runs.")
	parser.add_argument("--debug", action="store_true", default=False, help="If a PyCharm remote debug server should be started.")

	train_parser.add_argument("-m", "--model", default=default_model, help=f"The HuggingFace model you want to use. Default is \"{default_model}\".")
	train_parser.add_argument("-mr", "--model-revision", type=str, default=None, help="The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("-d", "--dataset", default=None, help=f"The HuggingFace dataset you want to use. Default is \"{default_dataset}\".")
	train_parser.add_argument("-dr", "--dataset-revision", type=str, default=None, help="The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("-p", "--prune", type=float, default=0.2, help="The amount of parameters to prune.")
	train_parser.add_argument("-n", "--name", type=str, default=f"run_{dt.now().isoformat()}", help="The name of the experiment.")

	evaluate_parser.add_argument("-m", "--model", type=str, default=None, help="Only experiments based on this model will be evaluated.")
	evaluate_parser.add_argument("-d", "--dataset", default=default_dataset, help="The dataset that will be used to evaluate the models.")
	evaluate_parser.add_argument("-me", "--max-evaluations", type=int, default=None, help="The maximum number of rows to look at during the evaluation. Default is no limit.")
	evaluate_parser.add_argument("-sf", "--sql-filter", type=str, default=None, help="Only experiments based on this SQL condition will be evaluated. i.e. `dataset = 'glue'`.")
	evaluate_parser.add_argument("-mt", "--max-new-tokens", type=int, default=256, help="The maximum amount of new tokens each model is allowed to add to the response.")
	evaluate_parser.add_argument("-bl", "--baseline", type=str, default=None,
								 help="The model to highlight as the baseline. If an integer is given, then the database will search for the row with this ROWID. If a string is given, it will use the given directory or HF Hub model.")
	evaluate_parser.add_argument("-mf", "--mask-filter", type=str, default=None, choices=SELECTOR_LOOKUP.keys())
	evaluate_parser.add_argument("-a", "--ablation", default=False, action="store_true", help="If the ablation graphics should be generated.")
	# Normally, I would add more SQL injection protection for this, but I doubt you would harm your own SQLite database

	dataset_parser.add_argument("-r", "--repo-id", default="ArxivCap", help="The namespace of the uploaded dataset.")
	dataset_parser.add_argument("-m", "--message", help="The commit message for uploading this dataset.")

	args = parser.parse_args(args)

	login(token=args.token)
	folder: Path = args.folder.resolve()
	logger = create_logger(folder, args.log)
	connection = get_connection(args.sqlite3_file)

	try:
		if args.debug:
			import pydevd_pycharm
			pydevd_pycharm.settrace("localhost", port=5952, stdout_to_server=True, stderr_to_server=True, suspend=False)

		match args.command:
			case "train":
				model = HFObject(args.model, args.model_revision)
				if args.dataset is not None:
					dataset = HFObject(args.dataset, args.dataset_revision)
				else:
					dataset = None
				start_time = dt.now()

				safetensor_file, pretrained_file = training_pipeline(model, folder, args.prune, dataset, logger=logger, seed=args.seed)
				with safe_cursor(connection) as cursor:
					cursor.execute("INSERT INTO experiments(dataset, base_model, pretrained_model, \"name\", experiment_directory, tokenizer_directory, pipeline_start) VALUES (:dataset, :base_model, :pretrained_model, :name, :directory, :tokenizer_dir, :start_time)",
								   dict(
										dataset=args.dataset,
										base_model=args.model,
										pretrained_model=str(pretrained_file),
										name=args.name,
										directory=str(folder),
										tokenizer_dir=str(folder / "tokenizer"),
										start_time=start_time
								   ))
			case "evaluate":
				if args.mask_filter is None:
					mask_selector = None
				else:
					if (mask_selector := SELECTOR_LOOKUP.get(args.mask_filter)) is None:
						from argparse import ArgumentError
						raise ArgumentError(f"\"{args.mask_filter}\" is not a valid mask selector option.")

				evaluation_pipeline(args.dataset, folder, connection, logger, max_evaluations=args.max_evaluations,
									sql_filter=args.sql_filter, baseline=args.baseline, mask_selector=mask_selector,
									ablation=args.ablation)
			case "dataset":
				from .generate_dataset import generate_dataset
				dataset = generate_dataset(args.repo_id, args.message, logger=logger)
	except BaseException as e:
		logger.exception("An error stopped the pipeline.", exc_info=e)
		raise e

	connection.commit()
	connection.close()
	# test(model, dataset, path)


if __name__ == "__main__":
	main()
