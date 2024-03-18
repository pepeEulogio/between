from model.auto_arima import AutoArima
from model.config import input_path
from loggings.logger_creator import logger_all
from datetime import datetime

if __name__ == "__main__":

    logger_all.info(f'{datetime.now()} | Proccess Started')

    # Object generation
    model = AutoArima(
        path_train=input_path,
        target_name='y',
        date_format='%d.%m.%y',
        freq='MS',
        steps=12
    )

    # Process execution
    model.pipeline()

    logger_all.info(f'{datetime.now()} | Process Finished')