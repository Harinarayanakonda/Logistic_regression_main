import logging
from src.config.settings import AppSettings

def setup_logging():
    logging.basicConfig(
        level=AppSettings.LOGGING_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()