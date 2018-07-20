import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

dataset = pd.read_csv("../datasets/encoded_data.csv")

x_train = dataset.drop("MonthlyIncome",axis =1)
y_train = dataset["MonthlyIncome"]

rfc.fit(x_train,y_train)

