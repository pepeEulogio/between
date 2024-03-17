from model.auto_arima import AutoArima
from model.config import input_path
from loggings.logger_creator import logger_all
from datetime import datetime

if __name__ == "__main__":
    logger_all.info(f'{datetime.now()} | Proccess Started')
    model = AutoArima(
        input_path, 'y','%d.%m.%y', 'MS', 12)

    model.pipeline()
    logger_all.info(f'{datetime.now()} | Process Finished')