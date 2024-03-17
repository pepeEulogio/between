from model.auto_arima import AutoArima

if __name__ == "__main__":
    path = 'data/train/train.csv'

    model1 = AutoArima(
        path, 'y','%d.%m.%y', 'MS', 12)

    model1.pipeline()

    print('Hello')