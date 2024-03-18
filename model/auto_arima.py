import pandas as pd
from skforecast.Sarimax import Sarimax
from sklearn.metrics import mean_absolute_percentage_error
from model.config import results_path, grid_sarima
import warnings
from datetime import datetime
from loggings.logger_creator import logger_all


class AutoArima:
    """
    It allows to launch predictions of a univariate time series, through Sarima models.
    The best arima is selected from a set of 10 configurations.
    """

    train = None
    model = None
    predictions = None
    train_grid = None
    test_grid = None
    mape_grip = None
    best_arima = None
    best_seasonality = None


    def __init__(self, path_train, target_name, date_format, freq, steps):
        """
        Input to create the target
        """
        self.path_train = path_train
        self.target_name = target_name
        self.date_format = date_format
        self.freq = freq
        self.steps = steps

    def __load_data(self):
        """
        Load data
        """
        logger_all.info(f'{datetime.now()} | Load data')
        self.train = pd.read_csv(self.path_train)

    def __preprocessing_data(self):
        """
        Adapt the input to the format required by the model
        """
        logger_all.info(f'{datetime.now()} | Preprocessing data')
        self.train.rename(columns={self.train.columns[0]: 'date'}, inplace=True)
        self.train = self.train.dropna().copy()
        self.train['date'] = pd.to_datetime(self.train['date'], format=self.date_format)
        self.train.set_index('date', inplace=True)
        self.train = self.train.asfreq(self.freq)
        self.train = self.train[self.target_name]

    def __grid_arima(self):
        """
         Generate a dictionary with the metrics (MAPE) calculated after backtesting the last 6 months of input for
         each model in the set.
        """
        logger_all.info(f'{datetime.now()} | Grid search')
        warnings.filterwarnings("ignore")
        test_grid = self.train.tail(6).copy()
        train_grid = self.train.drop(self.train.tail(6).index)
        self.mape_grip = {}
        for key in grid_sarima.keys():
            order = grid_sarima[key]['order']
            seasonal_order = grid_sarima[key]['seasonal_order']

            model = Sarimax(order=order, seasonal_order=seasonal_order)
            model.fit(y=train_grid)

            predictions = model.predict(steps=6)

            self.mape_grip[key] = mean_absolute_percentage_error(test_grid, predictions)
            logger_all.info(f'{datetime.now()} | Grid search | {key} | mape: {self.mape_grip[key]}')

    def __select_best_arima(self):
        """
         Generate a dictionary with the metrics (MAPE) calculated after backtesting the last 6 months of input for
         each model in the set.
        """
        logger_all.info(f'{datetime.now()} | Select best arima')
        best_sarima = min(self.mape_grip, key=self.mape_grip.get)
        self.best_arima = grid_sarima[best_sarima]['order']
        self.best_seasonality = grid_sarima[best_sarima]['seasonal_order']
        logger_all.info(f'{datetime.now()} | Select best arima | Arima: {self.best_arima} | Seasonality: {self.best_seasonality}')


    def __fit_arima(self):
        """
        Select the configuration that has the best metrics
        """
        logger_all.info(f'{datetime.now()} | Fit arima')
        self.model = Sarimax(order=self.best_arima, seasonal_order=self.best_seasonality)
        self.model.fit(y=self.train)

    def __predictions(self):
        """
        Generate predictions
        """
        logger_all.info(f'{datetime.now()} | Prediction')
        self.predictions = self.model.predict(steps=self.steps)

    def __save_results(self):
        """
        Save results
        """
        logger_all.info(f'{datetime.now()} | Save results')
        self.predictions.rename(columns={'pred': self.target_name}, inplace=True)
        self.predictions.to_csv(results_path)

    def pipeline(self):
        """
        Process execution pipeline
        """
        self.__load_data()
        self.__preprocessing_data()
        self.__grid_arima()
        self.__select_best_arima()
        self.__fit_arima()
        self.__predictions()
        self.__save_results()
