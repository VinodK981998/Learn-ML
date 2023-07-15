import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/Roadmap/Regression/canada_per_capita_income.csv")
df = df.dropna()

%matplotlib inline
plt.xlabel('Year')
plt.ylabel('Price')
plt.scatter(df.year,df['per capita income (US$)'],color = 'red', marker = '+')

x_train, x_test, y_train, y_test = (train_test_split(df['year'], df['per capita income (US$)'], test_size = 0.25))

linreg = linear_model.LinearRegression()
x_train = x_train.values.reshape(-1, 1)
y_train = y_train.values.reshape(-1, 1)
x_test = x_test.values.reshape(-1, 1)
y_test = y_test.values.reshape(-1, 1)


linreg.fit(x_train,y_train)
print(linreg.score(x_test,y_test))
linreg.predict([[2012]])

#multivariatet variables 
linreg.coef_
