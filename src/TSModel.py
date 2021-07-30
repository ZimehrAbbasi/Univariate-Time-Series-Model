import pandas as pd
from Domain import Domain
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from matplotlib import pyplot
from collections import deque
from datetime import datetime, timedelta
import torch
from torch.autograd import Variable
from LinearRegression import linearRegression

# Working with data with the least count being days, i.e. every data point is of a new day


class TSModel:

    def __init__(self, freq, p, Train_path, Test_path=None, learning_rate=0.000000000005, ):
        self.freq = freq  # Possible Weeks(W), Months(M), Years(Y)
        self.model = linearRegression(p, 1)
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=learning_rate)
        self.X_train = []
        self.Y_train = []
        self.y_pred = []
        self.p = p

        if str(type(Train_path)) == "<class 'pandas.core.frame.DataFrame'>":
            self.Data_train = Train_path
        else:
            self.Data_train = pd.read_csv(Train_path)

        if str(type(Test_path)) == "<class 'pandas.core.frame.DataFrame'>":
            if str(type(Train_path)) == "<class 'pandas.core.frame.DataFrame'>":
                self.Data_test = Test_path
            else:
                self.Data_test = pd.read_csv(Test_path)
            self.X = self.Data_train
            self.X_test = self.Data_test
        else:
            self.X = self.Data_train
            self.X_test = None

        if self.freq == "D":
            self.Domain = Domain(1)
        elif self.freq == "W":
            self.Domain = Domain(7)
        elif self.freq == "M":
            self.Domain = Domain(31)
        elif self.freq == "Y":
            self.Domain = Domain(366)

    def split(self):

        n = self.Data_train.shape[0]
        self.X = self.Data_train.iloc[:round(n*0.8)]
        self.X_test = self.Data_train.iloc[round(n*0.8):]

    def create_domain(self):

        values = self.X.values.tolist()
        for val in values:
            self.Domain.add_value(val[0])

    def create_training_set(self, period):

        training_set = [[] for _ in range(self.Domain.bucket_size)]
        X_train = []
        Y_train = []

        bucket = self.Domain.head

        while bucket != None:

            for i, val in enumerate(bucket.values):
                training_set[i].append(val)

            bucket = bucket.next

        for sample in training_set:

            if len(sample) != self.Domain.size:
                sample.append(sum(sample)/len(sample))

            X_train.append(deque(sample[:period]))

            for val in sample[period:]:
                Y_train.append(val)

        return X_train, Y_train

    def train(self, period):

        if str(type(self.X_test)) != "<class 'pandas.core.frame.DataFrame'>":
            self.split()

        self.create_domain()
        self.X_train, self.Y_train = self.create_training_set(period)

        i = 1
        while i * len(self.X_train) <= len(self.Y_train):
            self.model.fit(np.array(self.X_train), np.array(
                self.Y_train[(i-1)*len(self.X_train):i*len(self.X_train)]))
            i += 1
            for j, val in enumerate(self.Y_train[(i-1)*len(self.X_train):i*len(self.X_train)]):
                self.X_train[j].append(val)
                self.X_train[j].popleft()

    def train_nn(self):

        if str(type(self.X_test)) != "<class 'pandas.core.frame.DataFrame'>":
            self.split()

        self.create_domain()
        self.X_train, self.Y_train = self.create_training_set(self.p)
        print(len(self.X_train), len(self.Y_train))

        epochs = 1000

        i = 1
        while i * len(self.X_train) <= len(self.Y_train):

            # Training
            for epoch in range(epochs):
                if torch.cuda.is_available():
                    inputs = Variable(torch.from_numpy(self.X_train).cuda())
                    labels = Variable(torch.from_numpy(self.Y_train).cuda())
                else:
                    inputs = Variable(torch.from_numpy(
                        np.array(self.X_train, np.float32)))
                    labels = Variable(torch.from_numpy(
                        np.array(
                            self.Y_train[(i-1)*len(self.X_train):i*len(self.X_train)], np.float32)))

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                outputs = outputs.reshape(-1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            i += 1
            for j, val in enumerate(self.Y_train[(i-1)*len(self.X_train):i*len(self.X_train)]):
                self.X_train[j].append(val)
                self.X_train[j].popleft()

    def test_nn(self):

        with torch.no_grad():
            if torch.cuda.is_available():
                predicted = self.model(Variable(torch.from_numpy(
                    self.X_train).cuda())).cpu().data.numpy()
            else:
                predicted = self.model(
                    Variable(torch.from_numpy(np.array(self.X_train, np.float32)))).data.numpy()

            self.y_pred = predicted
            print(self.y_pred)

    def test(self):

        self.y_pred = self.model.predict(self.X_train)

        if len(self.y_pred) > len(self.X_test):

            mse = mean_squared_error(
                self.X_test, self.y_pred[:len(self.X_test)])
            r2 = r2_score(self.X_test, self.y_pred[:len(self.X_test)])
            mape = mean_absolute_percentage_error(
                self.X_test, self.y_pred[:len(self.X_test)])

        elif len(self.y_pred) < len(self.X_test):

            mse = mean_squared_error(
                self.X_test[:len(self.y_pred)], self.y_pred)
            r2 = r2_score(self.X_test[:len(self.y_pred)], self.y_pred)
            mape = mean_absolute_percentage_error(
                self.X_test[:len(self.y_pred)], self.y_pred)

        else:

            mse = mean_squared_error(self.X_test, self.y_pred)
            r2 = r2_score(self.X_test, self.y_pred)
            mape = mean_absolute_percentage_error(self.X_test, self.y_pred)

        print("########################################")
        print("Test results:")
        print("Mean Squared Error: %.2f" % mse)
        print("R^2 score: %.2f" % r2)
        print("Mean Absolute Percentage Error: %.2f" % mape)
        print("########################################")

    def predict(self, period):

        self.y_pred = np.array([])

        new_train = []

        for i in self.X_train:
            new_train.append(deque(i))

        for _ in range(period):
            print(len(new_train))
            pred = self.model(
                Variable(torch.from_numpy(np.array(new_train, np.float32)))).data.numpy()

            for i in range(len(new_train)):

                new_train[i].append(pred[i])
                new_train[i].popleft()
                self.Domain.add_value(pred[i])

            self.y_pred = np.concatenate((self.y_pred, pred), axis=None)

    def calculate_interval(self):

        start = self.Data_train.index[-1].date() + timedelta(days=1)
        end = self.Data_train.index[-1].date() + \
            timedelta(days=(len(self.y_pred)))

        return start, end

    def show(self):

        start, end = self.calculate_interval()

        date_rng = pd.date_range(
            start=start, end=end, freq='D')

        visualize = pd.DataFrame(
            self.y_pred, index=date_rng, columns=["y_pred"])
        visualize["y_actual"] = self.X_test
        visualize["error"] = (visualize["y_pred"] -
                              visualize["y_actual"]).abs()
        print(self.Data_train[self.Data_train.columns[0]])
        ax = self.Data_train[self.Data_train.columns[0]
                             ].plot(color='r')
        am = visualize["y_pred"].plot(ax=ax, color='b')
        am.fill_between(visualize.index, visualize["y_pred"] - visualize["error"].mean(),
                        visualize["y_pred"] + visualize["error"].mean(), alpha=0.35)

        pyplot.show()
