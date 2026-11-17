"""Small file-loading helpers shared by the input adapter."""

import logging

import pandas as pd


logger = logging.getLogger(__name__)


def load_excel_file(file_path):
    """Load one Excel input and return ``None`` after a readable error."""
    try:
        return pd.read_excel(file_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        logger.error("Could not load Excel input %s: %s", file_path, exc)
        return None
