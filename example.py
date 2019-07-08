import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

train_path = "train.csv"
train = pd.read_csv(train_path)
test_path = "test.csv"
test = pd.read_csv(test_path)
print(train.head())
print(train.describe())
print(train.shape)
print((train["winner"] == 'red').value_counts(normalize=True))

target = train["winner"].values

features_forest = train[
    ["adc_blue", "adc_red", "jungle_blue", "jungle_red", "mid_blue", "mid_red", "supp_blue", "supp_red", "top_blue",
     "top_red"]]

test_features = test[
    ["adc_blue", "adc_red", "jungle_blue", "jungle_red", "mid_blue", "mid_red", "supp_blue", "supp_red", "top_blue",
     "top_red"]]

le = LabelEncoder()
le.fit(np.unique(features_forest))
features_converted = np.array([le.transform(samp) for samp in features_forest.values])
test_converted = np.array([le.transform(samp) for samp in test_features.values])
print(features_converted)
print(test_converted)

forest = RandomForestClassifier(max_depth=45, min_samples_split=2, n_estimators=22, random_state=2)
my_forest = forest.fit(features_converted, target)
print(my_forest.score(features_converted, target))

my_prediction = my_forest.predict(test_converted)


print(my_prediction)

my_solution = pd.DataFrame(my_prediction, test["id"])

my_solution_exported = my_solution.to_csv("solution.csv")
