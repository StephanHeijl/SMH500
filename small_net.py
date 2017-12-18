from keras.layers import *
from keras.models import *
from keras.utils.np_utils import to_categorical
import sqlite3
import pandas
import random
from sklearn.metrics import *
import matplotlib.pyplot as plt

def build_model():
    inp = Input((TIME_LENGTH, 1))
    H = Dropout(0.1)(inp)
    H = LSTM(64, activation="selu", kernel_initializer="lecun_normal", return_sequences=True)(H)
    H = Dropout(0.2)(H)
    H = LSTM(64, activation="selu", kernel_initializer="lecun_normal")(H)
    """
    H = Dropout(0.2)(H)
    for i in range(4):
        H = Dense(128, activation="selu", kernel_initializer="lecun_normal")(H)
        H = Dropout(0.3)(H)
    """
    #H = Flatten()(inp)
    #H = Dense(64)(H)
    out = Dense(5, activation="softmax")(H)

    M = Model(inp, out)
    M.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return M


def get_data():
    conn = sqlite3.connect("coins.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM coins")
    data = cursor.fetchall()
    df = pandas.DataFrame(data, columns=["ts", "coin", "USD", "EUR", "BTC"])
    return df

data = get_data()
X, y = [], []
np.random.seed(0)
TIME_LENGTH = 20
TIME_GAP = 5
N_SAMPLES_PER_COIN = 50

X_test, y_test = [], []

coin_list = list(set(data.coin))
random.seed(0)
random.shuffle(coin_list)
test_coins = coin_list[-47:]
test_coins += ["BTC", "ETH", "DASH"]

def get_class(diff, std):
    if diff < -std * 0.4:
        return 0
    if diff < -std * 0.05:
        return 1
    if diff > std * 0.4:
        return 4
    if diff > std * 0.05:
        return 3

    return 2



for coin, data_points in data.groupby("coin"):

    #data_points.iloc[:, -3] -= data_points.iloc[:, -3].mean()
    #data_points.iloc[:, -3] /= (data_points.iloc[:, -3].std() + 1e-6)

    if len(data_points) < N_SAMPLES_PER_COIN:
        continue

    if coin not in test_coins:
        indices = np.random.choice(
            np.arange(0, 300 - TIME_LENGTH - TIME_GAP, 1),
            (N_SAMPLES_PER_COIN, ),
            replace=False
        )
        for x in sorted(indices):
            _x = data_points.iloc[x:x + TIME_LENGTH, -3].values
            try:
                data_points.iloc[x + TIME_LENGTH + TIME_GAP]
            except IndexError:
                continue
            future = data_points.iloc[x + TIME_LENGTH + TIME_GAP, -3]
            diff = _x[-5:].mean() - future
            _y = get_class(diff, _x.std())

            _x -= _x.mean()
            _x /= _x.std() + 1e-6

            X.append(_x)
            y.append(_y)

    # Fill test dataset

    if coin in test_coins:
        indices = np.random.choice(
            np.arange(300, 365 - TIME_LENGTH - TIME_GAP, 2),
            (N_SAMPLES_PER_COIN / 5, ),
            replace=False
        )

        for x in sorted(indices):
            _x = data_points.iloc[x:x+TIME_LENGTH, -3].values
            future = data_points.iloc[x+TIME_LENGTH+TIME_GAP, -3]
            diff = _x[-5:].mean() - future
            _y = get_class(diff, _x.std())

            _x -= _x.mean()
            _x /= _x.std() + 1e-6

            X_test.append(_x)
            y_test.append(_y)

X = np.array(X)
X = np.expand_dims(X, -1)

X_test = np.array(X_test)
X_test = np.expand_dims(X_test, -1)

print X.shape, X_test.shape

y = np.array(y, dtype="float32")
y = to_categorical(y)
y_test = np.array(y_test, dtype="float32")
y_test = to_categorical(y_test)

shuffled_indices = np.arange(X.shape[0])
np.random.shuffle(shuffled_indices)
X = X[shuffled_indices]
y = y[shuffled_indices]

model = build_model()
model.fit(X, y, epochs=10, shuffle=True, validation_split=0.1, batch_size=128)

print "Training"
y_pred = model.predict(X[:1000])
print classification_report(y[:1000], y_pred > 0.5, digits=4)

print "Validation"
y_pred = model.predict(X_test)
print classification_report(y_test, y_pred > 0.5, digits=4)

model.save("small_net.h5")

model = Model(inputs=model.input, outputs=model.layers[-2].output)

reduced = model.predict(X_test)
np.save("reduced.npy", reduced)
np.save("classes.npy", y_test)
np.save("preds.npy", y_pred)
np.save("timeseries.npy", X_test)
