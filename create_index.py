import pandas
import numpy as np

N_COINS_IN_INDEX = 50
TARGET_PERCENTAGE = 0.03
TARGET_PERIOD = 26
INVESTMENT = 1000
ITERATIONS = 10

coin_data = pandas.read_csv("coins.csv", index_col=0)

for i in range(ITERATIONS):
    period = np.random.choice(range(0, 26))
    coin_sample = coin_data.iloc[
        np.random.choice(range(len(coin_data.index)), 50, replace=False),
        [period, period + TARGET_PERIOD]
    ]
