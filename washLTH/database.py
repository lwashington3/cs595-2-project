import sqlite3

from contextlib import contextmanager
from datetime import datetime as dt
from pathlib import Path
from re import match
from typing import Generator

__all__ = ["get_connection", "safe_cursor"]


def adapt_datetime_iso(date: dt) -> str: # TODO: Test these
	return date.isoformat()


def convert_datetime(val: bytes) -> dt:
	return dt.fromisoformat(val.decode())


@contextmanager
def safe_cursor(connection: sqlite3.Connection) -> Generator[sqlite3.Cursor]:
	try:
		cursor = connection.cursor()
		yield cursor
		cursor.close()
		connection.commit()
	except sqlite3.OperationalError as e:
		connection.rollback()
		raise e


def get_connection(location: Path) -> sqlite3.Connection:
	db_already_exists = location.exists()
	connection = sqlite3.connect(location)
	connection.create_function("REGEXP", 2, lambda pattern, string: match(pattern, string) is not None)
	sqlite3.register_adapter(dt, adapt_datetime_iso)
	sqlite3.register_converter("timestamp", convert_datetime)

	if not db_already_exists:
		with safe_cursor(connection) as cursor:
			cursor.execute("""CREATE TABLE experiments(
								dataset TEXT DEFAULT NULL,
								base_model TEXT,
								pretrained_model TEXT DEFAULT NULL,
								\"name\" TEXT,
								experiment_directory TEXT,
								tokenizer_directory TEXT,
								pipeline_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
							)""")

	return connection
