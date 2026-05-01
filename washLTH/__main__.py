from .defaults import DEFAULT_SEED
from .hf_tools import HFObject, login
from .training import training_pipeline
from .evaluation import evaluation_pipeline
from .database import get_connection, safe_cursor

from pathlib import Path
from torch.cuda import is_available

import logging


old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
	record = old_factory(*args, **kwargs)

	if is_available():
		from torch.cuda import mem_get_info

		free, total = mem_get_info()

		record.free_memory = f"{free / 1e9:,.6f} GB"
		record.total_memory = f"{total / 1e9:,.6f} GB"
	return record


def create_logger(directory: Path, log_file: str) -> logging.Logger:
	from sys import stdout
	from torch.cuda import is_available
	import logging.config

	directory.mkdir(exist_ok=True, parents=True)
	logging.config.dictConfig({
		"version": 1,
		"disable_existing_loggers": True,
		"formatters": {
			"print": {
				"format": "%(asctime)s\t%(pathname)s:%(lineno)3d | Free: %(free_memory)s - Total: %(total_memory)s: %(message)s" if is_available() else "%(asctime)s\t%(pathname)s:%(lineno)3d: %(message)s",
				"datefmt": "%m/%d/%Y %H:%M:%S"
			},
			"file": {
				"format": "%(levelname)5s %(asctime)s.%(msecs)03d | PID: %(process)s | Thread: %(thread)d | %(name)s | Function: %(funcName)s() in %(pathname)s on line %(lineno)3d | Free: %(free_memory)s - Total: %(total_memory)s | %(message)s" if is_available() else "%(levelname)5s %(asctime)s.%(msecs)03d | PID: %(process)s | Thread: %(thread)d | %(name)s | Function: %(funcName)s() in %(pathname)s on line %(lineno)3d | %(message)s",
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
	default_dataset = "/lth/datasets/ArxivCap-Minimal/*.parquet"
	# default_dataset = "lwashington3/exsclaim-caption-distributor"

	parser = ArgumentParser(prog="washLTH")

	subparsers = parser.add_subparsers(dest="command", required=True)
	train_parser = subparsers.add_parser("train", help="Run the training pipeline.")
	evaluate_parser = subparsers.add_parser("evaluate", help="Run a pruned model.")
	dataset_parser = subparsers.add_parser("dataset", help="Pre-process the dataset and upload it to HuggingFace Hub.")

	# test_parser = train_parser.add_parser("test", help="Run a pruned model.")

	parser.add_argument("-t", "--token", default=None,
						help="Your HuggingFace token. If not given the system will pull it from the \"HF_TOKEN\" environment variable.")
	parser.add_argument("-f", "--folder", type=Path, default=Path("models") / f"run_{dt.now().isoformat()}",
						help="The folder where models and datasets related to this run are downloaded and stored.")
	parser.add_argument("-l", "--log", type=str, default="LTH.log", help="The name of the log directory.")
	parser.add_argument("-s", "--seed", type=int, default=DEFAULT_SEED, help="The seed of the pseudo random number generator.")
	parser.add_argument("-db", "--sqlite3-file", type=Path, default=Path(__file__).parent.parent / "LTH.db",
						help="The SQLite database file to hold information about experiments between runs.")

	train_parser.add_argument("-m", "--model", default=default_model, help=f"The HuggingFace model you want to use. Default is \"{default_model}\".")
	train_parser.add_argument("-mr", "--model-revision", type=str, default=None, help=f"The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("-d", "--dataset", default=default_dataset, help=f"The HuggingFace model you want to use. Default is \"{default_dataset}\".")
	train_parser.add_argument("-dr", "--dataset-revision", type=str, default=None, help=f"The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("-p", "--prune", type=float, default=0.2, help="The amount of parameters to prune.")
	train_parser.add_argument("-n", "--name", type=str, default=f"run_{dt.now().isoformat()}", help=f"The name of the experiment.")

	evaluate_parser.add_argument("-m", "--model", type=str, default=None, help=f"Only experiments based on this model will be evaluated.")
	evaluate_parser.add_argument("-d", "--dataset", default=default_dataset, help=f"The dataset that will be used to evaluate the models.")
	evaluate_parser.add_argument("-me", "--max-evaluations", type=int, default=None, help=f"The maximum number of rows to look at during the evaluation. Default is no limit.")
	evaluate_parser.add_argument("-sf", "--sql-filter", type=str, default=None, help=f"Only experiments based on this SQL condition will be evaluated. i.e. `dataset = 'glue'`.")
	# Normally, I would add more SQL injection protection for this, but I doubt you would harm your own SQLite database

	dataset_parser.add_argument("-r", "--repo-id", default="ArxivCap", help="The namespace of the uploaded dataset.")
	dataset_parser.add_argument("-m", "--message", help="The commit message for uploading this dataset.")

	args = parser.parse_args(args)

	login(token=args.token)
	folder: Path = args.folder.resolve()
	logger = create_logger(folder, args.log)
	connection = get_connection(args.sqlite3_file)

	try:
		match args.command:
			case "train":
				model = HFObject(args.model, args.model_revision)
				dataset = HFObject(args.dataset, args.dataset_revision)
				start_time = dt.now()

				safetensor_file = training_pipeline(model, dataset, folder, args.prune, logger=logger, seed=args.seed)
				with safe_cursor(connection) as cursor:
					cursor.execute("INSERT INTO experiments VALUES (:dataset, :base_model, :name, :safetensors, :tokenizer_dir, :start_time)",
								   dict(
										dataset=args.dataset,
										base_model=args.model,
										name=args.name,
										safetensors=str(safetensor_file),
										tokenizer_dir=str(folder / "tokenizer"),
										start_time=start_time
								   ))
			case "evaluate":
				evaluation_pipeline(args.dataset, args.folder, connection, logger, args.model, sql_filter=args.sql_filter)
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
