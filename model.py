import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


data = pd.read_csv("heart.csv")


# Outliers
def outliers(var):
    q1 = np.percentile(var, 25, axis=0)
    q3 = np.percentile(var, 75, axis=0)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr
    return var[(var > low_limit) & (var < up_limit)]


# Numerical features
data_f = []
num = [var for var in data.columns if data[var].nunique() > 4]
for variables in num:
    data_f.append(outliers(data[variables]))
data_df = (pd.DataFrame(data_f)).T
data_df.fillna(data_df.median(), inplace=True)

# Categorical features
categorical = np.array([var for var in data.columns if data[var].nunique() <= 4])

categorical_df = pd.DataFrame(data[categorical])


# log normal on features
def log_normal(var):
    data_df[var] = np.random.lognormal(data_df[var].mean(), data_df[var].std(), data_df[var].shape[0])
    data_df[var], m = stats.boxcox(data_df[var])


for variable in data_df.columns:
    if variable != 'chol':
        log_normal(variable)

df = pd.concat([data_df, categorical_df], axis=1)

# respective independent and dependant features
X = df.drop(columns=['target'])
y = df['target']

# splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

random_classifier = RandomForestClassifier(n_estimators=100, criterion='gini')

random_classifier.fit(X_train, y_train)

pickle.dump(random_classifier, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))
