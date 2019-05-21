import pandas as pd
import os

HOUSING_PATH = "datasets\housing"
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
print(housing.info())
print(housing['ocean_proximity'].value_counts())
print(housing.describe())
import matplotlib.pyplot as plt
housing.hist(bins = 50, figsize=(20,15))
plt.show()
