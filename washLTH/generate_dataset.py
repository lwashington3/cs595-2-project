from .general import notify, get_number_of_processors

from io import BytesIO
from datasets import load_dataset, Dataset, IterableDataset, Image as DImage, Features
from pathlib import Path
from PIL import Image
from typing import Optional
from tqdm import tqdm


def has_subcaptions(row, index=None, **kwargs) -> bool:
	file = kwargs.get("file")
	pbar: Optional[tqdm] = kwargs.get("pbar")
	if pbar is not None:
		pbar.update(1)
		if index is not None:
			pbar.write(f"{index:,}", file=file)

	captions = row["caption_images"][0]
	pairs = captions["cil_pairs"]
	usable = any(map(lambda pair: bool(pair["sub_caption"]), pairs))

	if usable:
		message = f"Found usable row at index: {index:_}."
		if pbar is not None:
			pbar.write(message, file=file)
		elif file is not None:
			file.write(message)

	return usable


def decode_nested_image(row):
	for caption_image in row["caption_images"]:
		for cil_pair in caption_image["cil_pairs"]:
			img = cil_pair["image"]
			if not isinstance(img, Image.Image):
				continue

			buf = BytesIO()
			img.save(buf, format="PNG")
			cil_pair["image"] = {"bytes": buf.getvalue(), "path": None}
	return row


def write_parquet(dataset: IterableDataset, disk_loc: Path, pbar: tqdm, file=None):
	import pyarrow as pa
	import pyarrow.parquet as pq

	SHARD_SIZE = 100 # Rows per file
	writer = None
	shard_idx = 0
	row_count = 0
	batch_size = 10
	batch = [None] * batch_size
	idx = 0
	found = 0

	for row in dataset:
		batch[idx] = row
		found += 1
		pbar.write(f"Found rows: {found:,}", file=file)

		if idx + 1 == batch_size:
			table = pa.Table.from_pylist(batch)

			if writer is None:
				writer = pq.ParquetWriter(disk_loc / f"filtered_{shard_idx:04d}.parquet", table.schema)

			writer.write_table(table)
			idx = 0
			row_count += batch_size
			batch = [None] * batch_size
		else:
			idx += 1

		if writer is not None and row_count >= SHARD_SIZE:
			writer.close()
			writer = None
			shard_idx += 1
			row_count = 0

	if writer is not None:
		writer.close()


def generate_regular_dataset(data_parent: Path, skip=0):
	from datasets import load_dataset as base_load_dataset, Dataset
	from tqdm import tqdm

	processors = get_number_of_processors()

	original_dataset: Dataset = base_load_dataset("MMInstruction/ArxivCap", split="train", num_proc=processors)
	original_dataset.remove_columns(["src", "title", "abstract", "meta"])

	# features = original_dataset.features
	num_rows = original_dataset.info.splits.total_num_examples

	if skip:
		original_dataset = original_dataset.skip(skip)

	original_dataset = original_dataset.filter(has_subcaptions)
	pbar = tqdm(original_dataset, total=num_rows, desc="Filtering rows")

	if skip:
		pbar.update(skip)

	# made_through = 0
	# finished = False

	# dataset = Dataset.from_generator(original_dataset, gen_kwargs=dict(bar=pbar), features=features)
	original_dataset.to_parquet(data_parent / "solo.parquet", num_proc=processors)

	notify(f"Dataset pushed successfully to finished compiling!.")

	return original_dataset


def main(upload: bool = False):
	from multiprocessing import cpu_count

	my_repo = "lwashington3/ArxivCap"
	disk_loc = Path("/datasets/temp").resolve()
	disk_loc.mkdir(parents=True, exist_ok=True)

	# datasets = [load_from_disk(directory) for directory in directories]
	num_proc = cpu_count() - 2

	dataset = load_dataset("MMInstruction/ArxivCap", num_proc=None, split="train", streaming=True)
	dataset = dataset.cast_column("image", DImage(decode=False))

	if isinstance(dataset, Dataset):
		notify("Loaded dataset.")

	removal = ["src", "title", "abstract", "meta"]
	dataset = dataset.remove_columns(removal) # ["src", "title", "abstract", "meta"])
	if isinstance(dataset, Dataset):
		notify("Removed columns.")

	new_features = {k: v for k, v in dataset.features.items() if k not in set(removal)}

	pbar = tqdm(desc="Iterating through filtered data.", total=572_734, colour="green")
	dataset = dataset.filter(has_subcaptions, with_indices=True, fn_kwargs=dict(pbar=pbar))
	dataset = dataset.map(decode_nested_image)
	if isinstance(dataset, Dataset):
		notify("Filtered rows.")
	elif isinstance(dataset, IterableDataset):
		notify("Loaded dataset.")

		with open(disk_loc / "write.log", 'w') as f:
			write_parquet(dataset, disk_loc, pbar, file=f)

		notify("Should have written the files to storage.")

	if upload:
		from os import getenv
		from torch import Generator
		from torch.utils.data import random_split
		from .defaults import DEFAULT_SEED

		all_files = tuple(disk_loc.glob("*.parquet"))
		rng = Generator().manual_seed(DEFAULT_SEED)
		train_files, val_files, test_files = random_split(all_files, [0.7, 0.15, 0.15], generator=rng)

		dataset = load_dataset("parquet", data_files=dict(train=train_files, val=val_files, test=test_files),
							   streaming=True)

		# data_folder = disk_loc / "data"
		# data_folder.mkdir()
		#
		# for split, files in zip(
		# 	("train", train_files),
		# 	("val", val_files),
		# 	("test", test_files)
		# ):
		# 	# folder = data_folder / split
		# 	# folder.mkdir()
		# 	num_format = f"0{len(str(files))}d"
		# 	num_files = f"{len(files):{num_format}}"
		#
		# 	for i, file in enumerate(files):
		# 		file.move(data_folder / f"{split}-{i:{num_format}}-of-{num_files}.parquet")

		commit_info = dataset.push_to_hub(my_repo, private=True, token=getenv("HF_TOKEN"), num_proc=num_proc, # max_shard_size="500MB"
						commit_message="Filtered out the rows with non-null subcaptions from MMInstruction/ArxivCap.")
		notify(f"Dataset pushed successfully to {commit_info.commit_url}.")


if __name__ == "__main__":
	try:
		main()
		notify("The run succeeded.")
	except BaseException as e:
		notify(f"The run failed: {e}.", e)
		raise e
