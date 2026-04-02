from .hf_tools import HFObject, login
from .train import training_pipeline
from .pipeline import running_pipeline

from pathlib import Path

import logging


def test(base_model_namespace, dataset_namespace, experiment_directory: Path, push_to_hub=False):
	info = pull_model(model_name, path, dry_run=False)
	print(info)


old_factory = logging.getLogRecordFactory()


def record_factory(*args, **kwargs):
	from torch.cuda import mem_get_info

	free, total = mem_get_info()
	record = old_factory(*args, **kwargs)

	record.free_memory = f"{free / 1e9:,.6f} GB"
	record.total_memory = f"{total / 1e9:,.6f} GB"
	return record


def create_logger(directory: Path, log_file: str) -> logging.Logger:
	print_handler = logging.StreamHandler()
	print_handler.setFormatter(logging.Formatter("%(asctime)s\t%(pathname)s:%(lineno)3d | Free: %(free_memory)s - Total: %(total_memory)s: %(message)s", "%m/%d/%Y %H:%M:%S"))

	file_handler = logging.FileHandler(directory / log_file, mode="a")
	file_handler.setFormatter(logging.Formatter("%(levelname)5s %(asctime)s.%(msecs)03d | PID: %(process)s | Thread: %(thread)d | %(name)s | Function: %(funcName)s() in %(pathname)s on line %(lineno)3d | Free: %(free_memory)s - Total: %(total_memory)s | %(message)s",
	                          "%m/%d/%Y %H:%M:%S"))

	logging.setLogRecordFactory(record_factory)
	logging.basicConfig(level=logging.DEBUG, handlers=(print_handler, file_handler), force=True)
	return logging.getLogger(__name__)


def main(args=None):
	from argparse import ArgumentParser
	from datetime import datetime as dt

	default_model = "meta-llama/Llama-3.1-8B-Instruct"
	default_dataset = "lwashington3/exsclaim-caption-distributor"

	parser = ArgumentParser(prog="washLTH")

	subparsers = parser.add_subparsers(dest="command", required=True)
	train_parser = subparsers.add_parser("train", help="Run the training pipeline.")
	run_parser = subparsers.add_parser("run", help="Run a pruned model.")
	dataset_parser = subparsers.add_parser("dataset", help="Pre-process the dataset and upload it to HuggingFace Hub.")

	# test_parser = train_parser.add_parser("test", help="Run a pruned model.")

	parser.add_argument("--token", "-t", default=None, help="Your HuggingFace token. If not given the system will pull it from the \"HF_TOKEN\" environment variable.")
	parser.add_argument("--folder", "-f", type=Path, default=Path("models") / f"run_{dt.now().isoformat()}", help="The folder where models and datasets related to this run are downloaded and stored.")
	parser.add_argument("-l", "--log", type=str, default="LTH.log", help="The name of the log directory.")

	train_parser.add_argument("--model", "-m", default=default_model, help=f"The HuggingFace model you want to use. Default is \"{default_model}\".")
	train_parser.add_argument("--model-revision", "-mr", type=str, default=None, help=f"The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("--dataset", "-d", default=default_dataset, help=f"The HuggingFace model you want to use. Default is \"{default_dataset}\".")
	train_parser.add_argument("--dataset-revision", "-dr", type=str, default=None, help=f"The specific revision/tag of the version of the dataset to use.")
	train_parser.add_argument("--prune", "-p", type=float, default=0.2, help="The amount of parameters to prune.")

	dataset_parser.add_argument("-r", "--repo-id", default="ArxivCap", help="The namespace of the uploaded dataset.")
	dataset_parser.add_argument("-m", "--message", help="The commit message for uploading this dataset.")

	args = parser.parse_args(args)

	login(token=args.token)
	folder: Path = args.folder.resolve()
	logger = create_logger(folder, args.log)

	match args.command:
		case "train":
			model = HFObject(args.model, args.model_revision)
			dataset = HFObject(args.dataset, args.dataset_revision)

			# Checks if the directory is empty
			# if folder.exists():
			# 	files = folder.rglob("*")
			# 	if next(files, None) is not None:
			# 		raise IsADirectoryError(f"The directory {folder} exists and has files in it. Please provide an empty or non-existent directory.")
			# 	del files

			training_pipeline(model, dataset, folder, args.prune, logger=logger)
		case "run":
			running_pipeline(folder)
		case "dataset":
			from .dataset_loader import generate_dataset
			dataset = generate_dataset(args.repo_id, args.message, logger=logger)

	# test(model, dataset, path)


if __name__ == "__main__":
	main()
