'''
***** Final Project 2021 - Software Carpentry *****
Contributors - Mahin Gadkari , Rashi Sultania
Aim:- The aim of the code is to predict the likelyhood of forest fires
given the weather conditions.
The code uses 4 algorithms to predict forest fire. The user can chose either or all of
the algorithms to predict fire. 
'''

from functions import get_inputs
from functions import get_model_type
from functions import Model_train
from functions import Model_score
from functions import Model_predict
# Importing various models to predict the forest fire

print("************         Welcome to the Forest Fire Predictor        ************")
print("This predictor works on various machine learning classification\n" +
      "algorithms such as Decission Tree, Support Vector Machine, Multi Level Perceptron \n" +
      "and Gaussian Naive Bayes.\n")
print("To select which alogrithm to use  for the prediction use the given legend and \n" +
      "input the code when propted for model type:\n")
print("Model: Decision Tree Classifier                                    Code: DT\n" +
      "Model: Support Vector Classifier                                   Code: SVC\n" +
      "Model: Multi Level Perceptron Classifier                           Code: MLP\n" +
      "Model: Gaussian Naive Bayes Classifier                             Code: GNB\n" +
      "Model: The concensus between all models gives us the prediction    Code: ALL\n")
# Creating the interface for user to choose the model
model_type = get_model_type()
while True:
    Train = input("Do you want to retrain the model?(Y/N) : ")  # Giving options to the user to change the model
    if Train == "Y":
        Model_train(model_type)
        break
    elif Train == "N":
        break
    else:
        print("Enter a valid answer!")
while True:
    Score = input("Do you want to know the accuracy of the model?(Y/N) : ") # Asking for accuracy
    if Score == "Y":
        Model_score(model_type)
        break
    elif Score == "N":
        break
    else:
        print("Enter a valid answer!")
while True:
    Predict = input("Do you want to make a prediction?(Y/N) : ")  # Giving choice to user for forest fire prediction
    if Score == "Y":
        print("Enter the required data as requested")
        inputs = get_inputs()
        prediction = Model_predict(model_type, X=inputs) # Chosing either or all of the models
        break
    elif Score == "N":
        break
    else:
        print("Enter a valid answer!")

Model_score(model_type)
prediction = Model_predict(model_type, X=inputs)  # calling the function for fire prediction
print(prediction)   # Printing our final value
