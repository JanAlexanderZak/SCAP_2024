""" Collection of handy utility functions.
"""
import datetime as dt
import pytz


def log_msg(string: str) -> str:
    """ Prints out a [LOG] message with a current timestamp.

    Args:
        string (str): String to be printed out.
    """
    ISO_TIME = dt.datetime.now(tz=pytz.timezone("Europe/Berlin")).replace(microsecond=0).isoformat()
    string_msg = f"[LOG {ISO_TIME[:-6]}] {string}"
    print(string_msg)
    return string_msg
