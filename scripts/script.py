import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("HR_comma_sep.csv")

dataset["salary"] = LabelEncoder().fit_transform(dataset["salary"])

salary = np.add(dataset["salary"],1)
years = dataset["time_spend_company"]
last_eval = dataset["last_evaluation"]
projects = dataset["number_project"]

dataset["ROI"] = np.divide((np.multiply(last_eval,projects)),(np.multiply(salary,years)))

dataset.to_csv("data_with_ROI.csv")