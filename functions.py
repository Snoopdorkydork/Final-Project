'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import scipy
import numpy as np
import pyglet


def get_inputs():
    while True:
        temp = input('Please enter Temperature in Celcius : ')
        try:
            temp = float(temp)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        RH = input(
            'Please enter Reletive Humidity (RH) percentage value : ')
        try:
            RH = float(RH)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        wind = input('Please enter Wind speed in km/h : ')
        try:
            wind = float(wind)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        rain = input('Please enter rain in mm : ')
        try:
            rain = float(rain)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        FFMC = input(
            'Please enter Fine Fuel Moisture Code (FFMC) index : ')
        try:
            FFMC = float(FFMC)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        DMC = input(
            'Please enter Duff Moisture Code (DMC) index : ')
        try:
            DMC = float(DMC)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        ISI = input(
            'Please enter Initial Spread Index (ISI) : ')
        try:
            ISI = float(ISI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        BUI = input(
            'Please enter Buildup Index (BUI) : ')
        try:
            BUI = float(BUI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        FWI = input(
            'Please enter Fire Weather Index (FWI) : ')
        try:
            FWI = float(FWI)
            break
        except ValueError:
            print("Enter a valid value!")
    while True:
        DC = input(
            'Please enter Drought Code (DC) index : ')
        try:
            DC = float(DC)
            break
        except ValueError:
            print("Enter a valid value!")

    in_val = np.array([[temp, RH, wind, rain, FFMC, DMC, DC, ISI, BUI, FWI]])
    return in_val


def get_model_type():
    while True:
        model_type = input("Please enter the model type: ")
        if model_type in ["DT", "SVC", "MLP", "GNB", "ALL"]:
            break
        else:
            print("Enter valid model code!")
    return model_type


def display_output(prediction):
    '''
    '''
    fire_file = "Output/spongebob.gif"
    not_fire_file = "Output/diablo.gif"

    fire_animation = pyglet.resource.animation(fire_file)
    not_fire_animation = pyglet.resource.animation(not_fire_file)

    if prediction[0] == 1:
        sprite = pyglet.sprite.Sprite(fire_animation)
        win = pyglet.window.Window(
            width=sprite.width, height=sprite.height)
        print("Prediction: FIRE")

    elif prediction[0] == 0:
        sprite = pyglet.sprite.Sprite(not_fire_animation)
        win = pyglet.window.Window(
            width=sprite.width, height=sprite.height)
        print("Prediction: NO FIRE")

    return win, sprite


def Model_train(model_type, filepath='Data/forestfires.csv'):
    df_class = pd.read_csv(filepath)

    df_class.columns = df_class.columns.str.replace(' ', '')

    df_class.dropna(axis="index", how="all", inplace=True)

    columns_sub = ["Classes", "Temperature", "RH", "Ws",
                   "Rain", "FFMC", "DMC", "DC", "ISI", "BUI", "FWI"]
    df_class.dropna(axis="index", how="any", inplace=True, subset=columns_sub)

    X = df_class.drop(["Classes", "day", "month", "year"], axis="columns")
    Y = df_class["Classes"].values
    X = X.astype(float)

    count = 0
    Y_label = []

    for i in Y:
        count += 1
        if i in ['fire', 'fire ', 'fire   ']:
            Y_label.append(1)
        elif i in [
            'not fire', 'not fire ', 'not fire   ',
                'not fire    ', 'not fire     ']:
            Y_label.append(0)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y_label, test_size=0.2)

    X_test_file = "Data/X_Test_data.pik"
    Y_test_file = "Data/Y_Test_data.pik"

    pickle.dump(X_test, open(X_test_file, 'wb'))
    pickle.dump(Y_test, open(Y_test_file, 'wb'))

    if model_type == "DT":
        model = tree.DecisionTreeClassifier()
        model = model.fit(X_train, Y_train)
        filename = "Data/Decision_Tree_Classifier_model.pik"
        pickle.dump(model, open(filename, 'wb'))

    elif model_type == "SVC":
        model = svm.SVC()
        model = model.fit(X_train, Y_train)
        filename = "Data/Support_Vector_Classifier_model.pik"
        pickle.dump(model, open(filename, 'wb'))

    elif model_type == "MLP":
        model = MLPClassifier(random_state=1, max_iter=700)
        model = model.fit(X_train, Y_train)
        filename = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        pickle.dump(model, open(filename, 'wb'))

    elif model_type == "GNB":
        model = GaussianNB()
        model = model.fit(X_train, Y_train)
        filename = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"
        pickle.dump(model, open(filename, 'wb'))

    elif model_type == "ALL":
        model_tree = tree.DecisionTreeClassifier()
        model_tree = model_tree.fit(X_train, Y_train)
        filename_tree = "Data/Decision_Tree_Classifier_model.pik"
        pickle.dump(model_tree, open(filename_tree, 'wb'))

        model_SVC = svm.SVC()
        model_SVC = model_SVC.fit(X_train, Y_train)
        filename_SVC = "Data/Support_Vector_Classifier_model.pik"
        pickle.dump(model_SVC, open(filename_SVC, 'wb'))

        model_MLP = MLPClassifier(random_state=1, max_iter=700)
        model_MLP = model_MLP.fit(X_train, Y_train)
        filename_MLP = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        pickle.dump(model_MLP, open(filename_MLP, 'wb'))

        model_GNB = GaussianNB()
        model_GNB = model_GNB.fit(X_train, Y_train)
        filename_GNB = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"
        pickle.dump(model_GNB, open(filename_GNB, 'wb'))

    else:
        return print("Enter valid model type!")


def Model_score(model_type):
    X_test = pickle.load(open("Data/X_Test_data.pik", 'rb'))
    Y_test = pickle.load(open("Data/Y_Test_data.pik", 'rb'))

    if model_type == "DT":
        filename = "Data/Decision_Tree_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        score = model.score(X_test, Y_test)
        per = score * 100
        return print(
            "The accuracy percentage of the model is " + str(per) + "%")

    elif model_type == "SVC":
        filename = "Data/Support_Vector_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        score = model.score(X_test, Y_test)
        per = score * 100
        return print(
            "The accuracy percentage of the model is " + str(per) + "%")

    elif model_type == "MLP":
        filename = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        score = model.score(X_test, Y_test)
        per = score * 100
        return print(
            "The accuracy percentage of the model is " + str(per) + "%")

    elif model_type == "GNB":
        filename = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        score = model.score(X_test, Y_test)
        per = score * 100
        return print(
            "The accuracy percentage of the model is " + str(per) + "%")

    elif model_type == "ALL":
        filename_tree = "Data/Decision_Tree_Classifier_model.pik"
        model_tree = pickle.load(open(filename_tree, 'rb'))
        score = model_tree.score(X_test, Y_test)
        per = score * 100
        print(
            "The accuracy percentage of the DT model is " + str(per) + "%")
        filename_SVC = "Data/Support_Vector_Classifier_model.pik"
        model_SVC = pickle.load(open(filename_SVC, 'rb'))
        score = model_SVC.score(X_test, Y_test)
        per = score * 100
        print(
            "The accuracy percentage of the SVC model is " + str(per) + "%")
        filename_MLP = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        model_MLP = pickle.load(open(filename_MLP, 'rb'))
        score = model_MLP.score(X_test, Y_test)
        per = score * 100
        print(
            "The accuracy percentage of the MLP model is " + str(per) + "%")
        filename_GNB = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"
        model_GNB = pickle.load(open(filename_GNB, 'rb'))
        score = model_GNB.score(X_test, Y_test)
        per = score * 100
        print(
            "The accuracy percentage of the GNB model is " + str(per) + "%")

    else:
        return print("Enter valid model type!")


def Model_predict(model_type, X):
    if model_type == "DT":
        filename = "Data/Decision_Tree_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        prediction = model.predict(X)
        return prediction

    elif model_type == "SVC":
        filename = "Data/Support_Vector_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        prediction = model.predict(X)
        return prediction

    elif model_type == "MLP":
        filename = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        prediction = model.predict(X)
        return prediction

    elif model_type == "GNB":
        filename = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"
        model = pickle.load(open(filename, 'rb'))
        prediction = model.predict(X)
        return prediction

    elif model_type == "ALL":
        filename_tree = "Data/Decision_Tree_Classifier_model.pik"
        filename_SVC = "Data/Support_Vector_Classifier_model.pik"
        filename_MLP = "Data/Multi_Level_Perceptron_Classifier_model.pik"
        filename_GNB = "Data/Gaussian_Naive_Bayes_Classifier_model.pik"

        model_tree = pickle.load(open(filename_tree, 'rb'))
        model_SVC = pickle.load(open(filename_SVC, 'rb'))
        model_MLP = pickle.load(open(filename_MLP, 'rb'))
        model_GNB = pickle.load(open(filename_GNB, 'rb'))

        prediction_tree = model_tree.predict(X)
        prediction_SVC = model_SVC.predict(X)
        prediction_MLP = model_MLP.predict(X)
        prediction_GNB = model_GNB.predict(X)

        prediction = [
            prediction_tree, prediction_SVC, prediction_MLP, prediction_GNB]

        return scipy.stats.mode(prediction)[0][0]

    else:
        return print("Enter valid model type!")


if __name__ == "__main__":
    # model_type = get_model_type()
    # inputs = get_inputs()
    model_type = "DT"
    inputs = [[29.0, 57.0, 18.0, 0, 65.7, 3.4, 7.6, 1.3, 3.4, 0.5]]
    Model_score(model_type)
    prediction = Model_predict(model_type, X=inputs)
    print(prediction)
    # print(inputs)
    win, sprite = display_output(prediction)

    @win.event
    def on_draw():
        win.clear()
        sprite.draw()
    pyglet.app.run()
