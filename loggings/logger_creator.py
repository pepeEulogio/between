import logging

def get_logger():
    logger = logging.getLogger(__name__)

    filehandler = logging.FileHandler(filename='loggings/logs/loggings.log')
    filehandler.setLevel(level=logging.INFO)

    logger.addHandler(filehandler)
    return logger

logger_all = get_logger()