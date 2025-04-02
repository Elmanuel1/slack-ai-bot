# Configure logging
import logging

import logging
import sys

from config.settings import AppSettings

class Logger:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.configure_logger()

    def configure_logger(self):
        logging.basicConfig(
            level=getattr(logging, self.settings.app.log_level.upper(), logging.INFO),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )


    def get_logger(self):
        return logging.getLogger()
