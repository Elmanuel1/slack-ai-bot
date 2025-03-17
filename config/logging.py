# Configure logging
import logging

import logging

class Logger:
    def __init__(self, settings):
        self.settings = settings
        self.configure_logger()

    def configure_logger(self):
        log_level = getattr(logging, self.settings.app.log_level.upper(), logging.INFO)

        logging.basicConfig(
            level=log_level,
            format="[%(asctime)s] %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

    def get_logger(self):
        return logging.getLogger()
