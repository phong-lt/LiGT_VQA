import logging

#format_str = '[%(asctime)s]-(%(levelname)s):: %(message)s'
format_str = '[%(asctime)s] %(message)s'

logging.basicConfig(format=format_str)


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger