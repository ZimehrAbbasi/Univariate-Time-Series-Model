import pandas as pd
from TSModel import TSModel
import yfinance as yf
import numpy as np

df = pd.read_csv("../data/DailyDelhiClimateTrain.csv")
df.index = pd.to_datetime(df["date"].values)
df = df.drop(columns=['date', "meanpressure", "humidity", "wind_speed"])

df_test = pd.read_csv("../data/DailyDelhiClimateTest.csv")
df_test.index = pd.to_datetime(df_test["date"].values)
df_test = df_test.drop(
    columns=['date', "meanpressure", "humidity", "wind_speed"])

model = TSModel("Y", df, df_test)
model.train()
model.test()
model.predict(3)
model.show()
