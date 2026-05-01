from os import getenv
from multiprocessing import cpu_count
from typing import Callable


__all__ = ["get_number_of_processors", "notify"]


def get_number_of_processors(leave_free: int = 2):
	"""Gets the number of processors available on the system and leaves"""
	cpus = cpu_count()
	if leave_free is not None and 0 <= leave_free < cpus:
		return cpus - leave_free

	return cpus


NTFY_LINK = getenv("NTFY_LINK", None)


def notify(message: str | Callable[[str], str], exception=None):
	"""Send a message to through NTFY if the environment variable `NTFY_LINK` is set."""
	from httpx import post
	if not NTFY_LINK:
		print(message, exception)
		return

	if isinstance(exception, BaseException):
		from traceback import format_exc
		exception = format_exc()

	if callable(message):
		message = message(exception)

	return post(NTFY_LINK, content=message).is_success
