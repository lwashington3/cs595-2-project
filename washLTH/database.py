import sqlite3

from contextlib import contextmanager
from pathlib import Path
from typing import ContextManager

__all__ = ["get_connection", "safe_cursor"]


@contextmanager
def safe_cursor(connection: sqlite3.Connection) -> ContextManager[sqlite3.Cursor]:
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

	if not db_already_exists:
		with safe_cursor(connection) as cursor:
			cursor.execute("""CREATE TABLE experiments(
								dataset TEXT,
								base_model TEXT,
								\"name\" TEXT,
								safetensors_file TEXT,
								tokenizer_directory TEXT,
								pipeline_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
							)""")

	return connection
