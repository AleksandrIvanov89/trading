import logging

class Logger():

    def __init__(self, in_filename="logs.log"):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Create console handler and set level to debug
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.INFO)
        # Create file handler and set level to debug
        self.fh = logging.FileHandler(in_filename, mode='a', encoding=None, delay=False)
        self.fh.setLevel(logging.WARNING)
        # Create formatter
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        # Add formatter to handlers
        self.ch.setFormatter(self.formatter)
        self.fh.setFormatter(self.formatter)
        # Add handlers to logger
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)


    def info(self, in_text):
        self.logger.info(in_text)


    def error(self, in_text="Error occurred"):
        self.logger.error(in_text, exc_info=True)


    def exception(self, in_text="Exception occurred"):
        self.logger.exception(in_text)


    def warning(self, in_text="Warning occurred"):
        self.logger.warning(in_text)

