'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
'''


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB


df_class = pd.read_csv('Data/forestfires.csv')
df_area = pd.read_csv('Data/forestfires_1.csv')

df_class.columns = df_class.columns.str.replace(' ', '')
df_area.columns = df_area.columns.str.replace(' ', '')

df_class.dropna(axis="index", how="all", inplace=True)
df_area.dropna(axis="index", how="all", inplace=True)

columns_sub = ["Classes", "Temperature", "RH", "Ws",
               "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
columns_sub_1 = ["area", "temp", "RH", "wind",
                 "rain", "FFMC", "DMC", "DC", "ISI"]

df_class.dropna(axis="index", how="any", inplace=True, subset=columns_sub)
df_area.dropna(axis="index", how="any", inplace=True, subset=columns_sub_1)

inputs = df_class.drop(["Classes", "day", "month", "year"], axis="columns")
print(inputs.columns.values)
target = df_class["Classes"].values
inputs = inputs.astype(float)

count = 0
target_label = []

for i in target:
    count += 1
    if i in ['fire', 'fire ', 'fire   ']:
        target_label.append(0)
    elif i in [
        'not fire', 'not fire ', 'not fire   ',
            'not fire    ', 'not fire     ']:
        target_label.append(1)

inputs_train, inputs_test, target_train, target_test = train_test_split(
    inputs, target_label, test_size=0.2)

model_tree = tree.DecisionTreeClassifier()
model_tree = model_tree.fit(inputs_train, target_train)
model_SVC = svm.SVC()
model_SVC = model_SVC.fit(inputs_train, target_train)
model_MLP = MLPClassifier(random_state=1,max_iter=700)
model_MLP = model_MLP.fit(inputs_train, target_train)
model_NB = GaussianNB()
model_NB = model_NB.fit(inputs_train, target_train)

input_area = df_area.drop(["month", "day", "area"], axis="columns")
target_area = df_area["area"].values

input_2_train, input_2_test, target_2_train, target_2_test = train_test_split(
    input_area, target_area, test_size=0.2)

model_reg = MLPRegressor(activation="logistic", max_iter=1000, learning_rate_init=0.01).fit(input_2_train, target_2_train)
# model_reg = model_reg.fit(input_2_train, target_2_train)

print(model_NB.score(inputs_test, target_test))
# print(model_reg.score(input_2_test,target_2_test))
# print(model_reg.score(input_2_train,target_2_train))
# print(model_reg.loss_)
# print(model_reg.best_loss_)
# print(model_reg.n_iter_)
# print(input_area.shape)
# print(len(target_area))
# print(input_2_train.shape)
# print(input_2_test.shape)
# print(len(target_2_train))
# print(len(target_2_test))
