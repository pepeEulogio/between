input_path = 'data/train/train.csv'

results_path = 'data/output/test.csv'

grid_sarima = {
    'sarima_101_6': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 1, 6)
    },
    'sarima_101_12': {
        'order': (1, 0, 1),
        'seasonal_order': (1, 0, 1, 12)
    },
    'sarima_111_6': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 6)
    },
    'sarima_111_12': {
        'order': (1, 1, 1),
        'seasonal_order': (1, 1, 1, 12)
    },
    'sarima_102_6': {
        'order': (1, 0, 2),
        'seasonal_order': (1, 0, 2, 6)
    },
    'sarima_102_12': {
        'order': (1, 0, 2),
        'seasonal_order': (1, 0, 2, 12)
    },
    'sarima_112_6': {
        'order': (1, 1, 2),
        'seasonal_order': (1, 1, 2, 6)
    },
    'sarima_112_12': {
        'order': (1, 1, 2),
        'seasonal_order': (1, 1, 2, 12)
    },
    'sarima_211_6': {
        'order': (2, 1, 1),
        'seasonal_order': (2, 1, 1, 6)
    },
    'sarima_211_12': {
        'order': (2, 1, 1),
        'seasonal_order': (2, 1, 1, 12)
    },
}