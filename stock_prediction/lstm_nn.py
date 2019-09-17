from pandas.plotting import register_matplotlib_converters
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np


# ----------------------------------------------------------------------- #
# A 9 layer Neural Network utilising LSTM layers to predict stock prices. #
#                                                                         #
# Datasets of Google's and Microsoft's historical stock data taken from:  #
#                           www.macrotrends.net                           #
# ----------------------------------------------------------------------- #

# Constant:
#   DAYS_BEFORE: How many days before a date are used for the prediction
#   COMPANY: Which companies stock gets analyzed
DAYS_BEFORE = 30
COMPANY = "MSFT"

register_matplotlib_converters()


def arrange_data(data, days):
    """
    Building the arrays containing the values for DAYS_BEFORE days before day T,
    as well as the value of the day after day T.
    """
    days_before = []
    next_day = []

    for i in range(len(data) - days - 1):
        days_before.append(data[i:(i + days)])
        next_day.append(data[i + days + 1])

    return days_before, next_day


data = np.loadtxt(COMPANY + ".csv", delimiter=",", skiprows=15, comments="#", usecols=4, dtype=None)
dates = np.loadtxt(COMPANY + ".csv", delimiter=",", skiprows=15, comments="#", usecols=0, dtype="datetime64")

data = data.reshape(-1, 1)

# Normalization using sklearn's MinMaxScaler,
# thereby transforming every value to be between 0 and 1
min_max_scaler = MinMaxScaler(feature_range=(0, 1))
stock_data = min_max_scaler.fit_transform(data)

days_before_val, next_day_val = arrange_data(stock_data, DAYS_BEFORE)

# Splitting in training and testing data (80/20)
days_before_train, days_before_test, next_day_train, next_day_test = train_test_split(days_before_val,
                                                                                      next_day_val,
                                                                                      test_size=0.2,
                                                                                      shuffle=False)
# Trying to load a saved model...
model = None
try:
    model = keras.models.load_model("LSTM_" + COMPANY + ".h5")
except:
    pass

# if there is none, training a new model.
if model is None:
    model = keras.models.Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(DAYS_BEFORE, 1)))
    model.add(keras.layers.Dropout(0.2))

    # Adding a second LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units=50, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))

    # Adding a third LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units=50, return_sequences=True))
    model.add(keras.layers.Dropout(0.2))

    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))

    # Adding the output layer
    model.add(keras.layers.Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["acc"])

    model.fit(x=[days_before_train], y=[next_day_train], batch_size=12, epochs=5, verbose=1)

    model.save("LSTM.h5")

# Predicting the values
train_predict = model.predict([days_before_train])
train_predict = min_max_scaler.inverse_transform(train_predict)

test_predict = model.predict([days_before_test])
test_predict = min_max_scaler.inverse_transform(test_predict)

# Plotting the original data and overlaying it with the predictions.
fig = plt.figure(1)
plt.plot(dates, data, label="Real data", color="blue", alpha=0.7)
plt.plot(dates[:len(train_predict)], train_predict, label="Prediction (training)", color="k")
plt.plot(dates[-len(test_predict):], test_predict, label="Predictions (test)", linestyle='dashed', color="red")
plt.title("Stock prediction for " + COMPANY)
plt.legend()
plt.show()
