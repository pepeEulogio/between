import pandas as pd

from skforecast.Sarimax import Sarimax

class AutoArima:

    train = None
    model = None
    predictions = None
    best_arima = None
    best_seasonality = None

    def __init__(self, path_train, target_name, date_format, freq, steps):
        self.path_train = path_train
        self.target_name = target_name
        self.date_format = date_format
        self.freq = freq
        self.steps = steps

    def __load_data(self):
        self.train = pd.read_csv(self.path_train)

    def __preprocessing_data(self):
        self.train.rename(columns={self.train.columns[0]: 'date'}, inplace=True)
        self.train = self.train.dropna().copy()
        self.train['date'] = pd.to_datetime(self.train['date'], format=self.date_format)
        self.train.set_index('date', inplace=True)
        self.train = self.train.asfreq(self.freq)
        self.train = self.train[self.target_name]

    def __grid_arima(self):
        pass

    def __fit_arima(self):
        self.model = Sarimax(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        self.model.fit(y=self.train)

    def __predictions(self):
        self.predictions = self.model.predict(steps=self.steps)

    def __save_results(self):
        self.predictions.rename(columns={'pred': 'y'}, inplace=True)
        self.predictions.to_csv('data/output/test.csv')

    def pipeline(self):
        self.__load_data()
        self.__preprocessing_data()
        self.__fit_arima()
        self.__predictions()
        self.__save_results()