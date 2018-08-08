import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("HR_comma_sep.csv")

dataset["salary"] = LabelEncoder().fit_transform(dataset["salary"])

salary = np.add(dataset["salary"],1)
years = dataset["time_spend_company"]
last_eval = dataset["last_evaluation"]
projects = dataset["number_project"]
sat_level = dataset["satisfaction_level"]

dataset["ROI"] = np.divide((np.multiply(last_eval,projects)),(np.multiply(salary,years)))
dataset["attrition_rate"] = np.divide(last_eval,sat_level)

dataset.to_csv("data_analysis2.csv")
