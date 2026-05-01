import huggingface_hub as hf
from dataclasses import dataclass
from datasets import load_dataset
from pathlib import Path
from typing import Optional


__all__ = ["HFObject", "login", "pull_model", "pull_dataset", "upload_dataset"]


@dataclass
class HFObject:
	"""A dataclass to hold relevant parts of an HF object (model, dataset, etc.) together."""
	name: str
	revision: Optional[str] = None


def login(*args, **kwargs):
	try:
		hf.login(*args, **kwargs)
	except (AttributeError, TypeError) as e:
		print(e)
		hf._login.login(*args, **kwargs)


def pull_dataset(dataset_namespace: str, parent_directory: Path, force_download: bool = False):
	if not hf.repo_exists(dataset_namespace):
		raise hf.errors.RepositoryNotFoundError(f"Model namespace {dataset_namespace} does not exist.")

	repo_type, author, dataset_name = hf.repo_type_and_id_from_hf_id(dataset_namespace)
	repo_type = repo_type or "dataset"
	parent_directory.mkdir(parents=True, exist_ok=True)

	hf.snapshot_download(repo_id=dataset_namespace, local_dir=parent_directory / model_name, force_download=force_download,
	                     repo_type=repo_type)


def upload_dataset(repo_id: str, data_files: list[str | Path] | dict[str, str | Path], **kwargs):
	"""
	>>> files_directory = Path(__file__).parent.resolve()
	>>> files = {"train": files_directory / "train.json", "valid": files_directory / "validation.json", "test": files_directory / "test.json"}
	>>> upload_dataset("test_dataset", files, private=True)
	"""
	if isinstance(data_files, dict):
		files = {key: str(value) for key, value in data_files.items()}
	else:
		files = tuple(map(str, data_files))

	dataset = load_dataset("json", data_files=files)
	hf.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
	dataset.push_to_hub(repo_id, **kwargs)


def pull_model(model_namespace: HFObject, parent_directory: Path = None, **kwargs):
	"""Pulls a model's files from HF Hub to the local file storage."""
	model, revision = model_namespace.name, model_namespace.revision
	if not hf.repo_exists(model):
		raise hf.errors.RepositoryNotFoundError(f"Model namespace {model} does not exist.")

	repo_type, author, model_name = hf.repo_type_and_id_from_hf_id(model)

	if parent_directory is not None:
		parent_directory.mkdir(parents=True, exist_ok=True)
		local_dir = parent_directory / model_name
	else:
		local_dir = None

	download = hf.snapshot_download(repo_id=model, local_dir=local_dir, repo_type=repo_type, revision=revision, **kwargs)

	if kwargs.get("dry_run", False):
		return download

	return Path(download).resolve()
