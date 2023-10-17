import pandas as pd
import numpy as np
df=pd.read_csv("C:/Users/Shilpa/Desktop/dataset/marks1.csv")
df.info ()
x = df['CIE'].values.reshape(-1,1)
y = df['SEE'].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split (x, y,random_state =0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
y_pred = lm.predict(x_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
g=y_test.reshape(21,)
h=y_pred.reshape(21,)
print(g)
print(h)
print(x_train)
print(x_test)

