import pandas as pd
from TSModel import TSModel
import yfinance as yf
import numpy as np

df = pd.read_csv("DailyDelhiClimateTrain.csv")
df.index = pd.to_datetime(df["date"].values)
df = df.drop(columns=['date', "meanpressure", "humidity", "wind_speed"])

df_test = pd.read_csv("DailyDelhiClimateTest.csv")
df_test.index = pd.to_datetime(df_test["date"].values)
df_test = df_test.drop(
    columns=['date', "meanpressure", "humidity", "wind_speed"])

# df = pd.read_csv("GlobalLandTemperatures_GlobalTemperatures.csv")
# df.index = pd.to_datetime(df["dt"].values)
# df = pd.DataFrame(df["LandAverageTemperature"],
#                   columns=['avg_temp'])

# ticker = "aapl"
# stock = yf.Ticker(ticker)
# df = stock.history("10y", "1d")
# df.index.name = None
# df = np.log(np.log(df[["Close"]]))


# def change(string):
#     string = str(string)
#     return string[:4] + "-" + string[4:6] + "-" + string[6:]


# def format(df):
#     lister = df.values.tolist()

#     for i in range(len(lister)):
#         lister[i] = change(lister[i])

#     return lister


# df = pd.read_csv("daily-total-female-births-CA.csv")
# df.index = pd.to_datetime(df["date"].values)
# df = df.drop(columns=["date"])

model = TSModel("Y", 3, df, df_test)
# Number of values in the past to take into consideration
model.train_nn()
model.test_nn()
# Number of periods in the future to predict
model.predict(period=3)
model.show()
