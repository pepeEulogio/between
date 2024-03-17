from model.auto_arima import AutoArima
from model.config import input_path

if __name__ == "__main__":

    model = AutoArima(
        input_path, 'y','%d.%m.%y', 'MS', 12)

    model.pipeline()