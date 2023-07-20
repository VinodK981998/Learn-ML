import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from word2number import w2n

df = pd.read_csv("D:/Roadmap/Regression/hiring.csv")
df.experience = df.experience.fillna(0)
df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].median())

df['experience'] = df['experience'].astype(str)
df['experience'] = df['experience'].apply(w2n.word_to_num)


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


reg.predict([[2,9,6]])





