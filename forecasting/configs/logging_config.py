import os
import sys

from loguru import logger

from forecasting import PROJECT_SRC_PATH


def setup_logger():
    LOG_PATH = os.path.join(PROJECT_SRC_PATH, "logs")
    os.makedirs(LOG_PATH, exist_ok=True)

    fmt = "{time} - {file.name} - {level} - {message}"

    logger.remove()
    logger.add(sys.stderr, format=fmt, level="INFO")
    logger.add(
        os.path.join(LOG_PATH, "root.log"), format=fmt, level="DEBUG", rotation="500 MB"
    )
